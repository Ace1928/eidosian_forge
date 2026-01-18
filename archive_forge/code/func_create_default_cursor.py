import re
from uuid import UUID as _python_UUID
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from ... import sql
from ... import util
from ...sql import sqltypes
def create_default_cursor(self):
    return self._dbapi_connection.cursor(buffered=True)