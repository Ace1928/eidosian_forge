import asyncio
import sqlite3
import json
import logging
import aiosqlite
from .defaults import SQLITE_THREADS, SQLITE_TIMEOUT
from .base_cache import BaseCache, CacheEntry
def borrow(self, timeout=None):
    if not self._ready:
        raise RuntimeError('Pool not prepared!')

    class PoolBorrow:

        def __init__(s):
            s._conn = None

        async def __aenter__(s):
            s._conn = await asyncio.wait_for(self._free_conns.get(), timeout)
            return s._conn

        async def __aexit__(s, exc_type, exc, tb):
            if self._stopped:
                await s._conn.close()
                return
            if exc_type is not None:
                await s._conn.close()
                s._conn = await self._new_conn()
            self._free_conns.put_nowait(s._conn)
    return PoolBorrow()