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
def result_processor(self, dialect, coltype):
    if dialect.native_datetime:
        return None
    else:
        return DATE.result_processor(self, dialect, coltype)