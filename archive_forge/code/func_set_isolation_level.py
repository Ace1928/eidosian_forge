from the connection pool, such as when using an ORM :class:`.Session` where
from working correctly.  The pysqlite DBAPI driver has several
import math
import os
import re
from .base import DATE
from .base import DATETIME
from .base import SQLiteDialect
from ... import exc
from ... import pool
from ... import types as sqltypes
from ... import util
def set_isolation_level(self, dbapi_connection, level):
    if level == 'AUTOCOMMIT':
        dbapi_connection.isolation_level = None
    else:
        dbapi_connection.isolation_level = ''
        return super().set_isolation_level(dbapi_connection, level)