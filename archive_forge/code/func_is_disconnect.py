import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
def is_disconnect(self, e, connection, cursor):
    if isinstance(e, self.dbapi.OperationalError) and 'no active connection' in str(e):
        return True
    return super().is_disconnect(e, connection, cursor)