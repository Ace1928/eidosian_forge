from contextlib import asynccontextmanager
from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
def character_set_name(self):
    return self._connection.character_set_name()