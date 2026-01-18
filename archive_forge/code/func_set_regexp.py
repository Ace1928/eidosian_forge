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
def set_regexp(dbapi_connection):
    dbapi_connection.create_function('regexp', 2, regexp, **create_func_kw)