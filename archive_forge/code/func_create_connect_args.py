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
def create_connect_args(self, url):
    arg, opts = super().create_connect_args(url)
    opts['factory'] = self._fix_sqlite_issue_99953()
    return (arg, opts)