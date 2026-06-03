"""
OpenClaw Memory Backend Client
==============================

Implements the same async interface as Mem0Client so the LOCOMO runner
can swap backends without modification.

Each user_id maps to its own OpenClaw `--agent <user_id>` workspace,
giving natural isolation. Sessions are written as Markdown files under
`memory/`, then `openclaw memory index --force` builds the vector index.

Interface (matches Mem0Client):
  await client.add(messages, user_id, timestamp=epoch) -> dict | None
  await client.search(query, user_id, top_k, score_debug) -> list[dict]
  await client.delete_user(user_id) -> bool
  await client.close()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


OPENCLAW_HOME = Path(os.environ.get("OPENCLAW_WORKSPACE", str(Path.home() / ".openclaw" / "workspace")))


def _agent_workspace(user_id: str) -> Path:
    return OPENCLAW_HOME / user_id


def _safe_user_id(user_id: str) -> str:
    """OpenClaw agent ids should avoid shell-special chars."""
    return user_id.replace("/", "_").replace(" ", "_")


class OpenClawClient:
    """OpenClaw memory backend exposing the Mem0Client interface."""

    def __init__(
        self,
        mode: str = "local",  # ignored; for API parity
        host: str | None = None,  # ignored
        api_key: str | None = None,  # ignored
        rpm: int = 600,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
        index_concurrency: int = 1,
        **kwargs: Any,
    ) -> None:
        self.mode = mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        # Per-user lock + dirty flag so we batch index calls.
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._user_dirty: dict[str, bool] = {}
        self._user_indexed: dict[str, bool] = {}
        self._user_turn_counter: dict[str, int] = {}
        self._index_sem = asyncio.Semaphore(max(1, index_concurrency))

    async def __aenter__(self) -> "OpenClawClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        # Final flush: index any dirty workspaces.
        for uid, dirty in list(self._user_dirty.items()):
            if dirty:
                await self._index_now(uid)

    # =========================================================================
    # Add
    # =========================================================================

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Append a chunk to a per-user file; mark workspace dirty.

        We do not run `openclaw memory index --force` on every add — that
        would dominate runtime. Instead we coalesce: indexing happens at
        flush() time (before search) or when close() is called.
        """
        uid = _safe_user_id(user_id)
        ws = _agent_workspace(uid)
        memdir = ws / "memory"
        memdir.mkdir(parents=True, exist_ok=True)

        lock = self._user_locks.setdefault(uid, asyncio.Lock())
        async with lock:
            # Match mem0's chunking granularity: write each add() call as its
            # own turn_NNNN.md so OpenClaw's builtin engine produces ~1 chunk
            # per file. mem0 uses CHUNK_SIZE=1 (one dialog turn per memory),
            # and benchmarks expect this granularity for fair comparison.
            turn_idx = self._user_turn_counter.get(uid, 0)
            self._user_turn_counter[uid] = turn_idx + 1
            content_path = memdir / f"turn_{turn_idx:05d}.md"

            ts_str = ""
            if timestamp is not None:
                try:
                    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    ts_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    ts_str = ""
            elif observation_date:
                ts_str = observation_date

            lines: list[str] = []
            if ts_str:
                lines.append(f"## {ts_str}")
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                lines.append(f"**{role}**: {content}")
            block = "\n".join(lines) + "\n"

            content_path.write_text(block, encoding="utf-8")

            self._user_dirty[uid] = True

        return {"results": [{"memory": "queued", "event": "ADD"}]}

    # =========================================================================
    # Index (manual flush)
    # =========================================================================

    async def flush(self, user_id: str) -> None:
        uid = _safe_user_id(user_id)
        if self._user_dirty.get(uid, False):
            await self._index_now(uid)

    async def _index_now(self, uid: str) -> None:
        async with self._index_sem:
            t0 = time.monotonic()
            cmd = ["openclaw", "memory", "index", "--agent", uid, "--force"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=600)
            except asyncio.TimeoutError:
                proc.kill()
                logger.error("openclaw memory index timed out for %s", uid)
                return
            elapsed = time.monotonic() - t0
            if proc.returncode != 0:
                logger.warning(
                    "openclaw memory index failed for %s (rc=%s): %s",
                    uid, proc.returncode, err.decode("utf-8", "replace")[-300:],
                )
            else:
                logger.info("indexed %s in %.1fs", uid, elapsed)
            self._user_dirty[uid] = False
            self._user_indexed[uid] = True

    # =========================================================================
    # Search
    # =========================================================================

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        uid = _safe_user_id(user_id)
        # Lazy flush: index dirty workspace before first search.
        if self._user_dirty.get(uid, False):
            await self._index_now(uid)

        for attempt in range(self.max_retries):
            try:
                results = await self._search_once(query, uid, top_k)
                # Normalise to mem0-compatible shape.
                normalised = []
                for r in results:
                    snippet = r.get("snippet", "") or r.get("memory", "")
                    entry: dict[str, Any] = {
                        "memory": snippet,
                        "score": r.get("score", 0),
                        "id": r.get("id", ""),
                    }
                    path = r.get("path")
                    if path:
                        entry["path"] = path
                    if score_debug and ("vectorScore" in r or "textScore" in r):
                        entry["score_debug"] = {
                            "combined_score": r.get("score", 0),
                            "semantic_score": r.get("vectorScore", 0),
                            "bm25_score": r.get("textScore", 0),
                        }
                    normalised.append(entry)
                normalised.sort(key=lambda x: x.get("score", 0), reverse=True)
                return normalised
            except Exception as exc:
                logger.warning(
                    "openclaw search attempt %d/%d failed (user=%s): %s",
                    attempt + 1, self.max_retries, uid, str(exc)[:200],
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        logger.error("openclaw search failed after %d attempts for user=%s", self.max_retries, uid)
        return []

    async def _search_once(self, query: str, uid: str, top_k: int) -> list[dict]:
        cmd = [
            "openclaw", "memory", "search", query,
            "--agent", uid,
            "--json",
            "--max-results", str(top_k),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("openclaw memory search timed out")
        if proc.returncode != 0:
            raise RuntimeError(f"openclaw memory search rc={proc.returncode}: {err.decode('utf-8', 'replace')[-200:]}")

        text = out.decode("utf-8", "replace")
        # Strip leading non-JSON warnings.
        idx = text.find("{")
        if idx < 0:
            return []
        payload = text[idx:]
        # Trailing log lines like "[memory] chunks_vec ..." may follow JSON.
        # Find the closing brace of the top-level object.
        depth = 0
        end = -1
        for i, ch in enumerate(payload):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return []
        try:
            data = json.loads(payload[:end])
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            return data.get("results", [])
        return []

    # =========================================================================
    # Delete
    # =========================================================================

    async def delete_user(self, user_id: str) -> bool:
        uid = _safe_user_id(user_id)
        ws = _agent_workspace(uid)
        try:
            if ws.exists():
                shutil.rmtree(ws)
            self._user_dirty.pop(uid, None)
            self._user_indexed.pop(uid, None)
            self._user_turn_counter.pop(uid, None)
            return True
        except Exception as exc:
            logger.warning("Failed to delete openclaw workspace for %s: %s", uid, exc)
            return False
