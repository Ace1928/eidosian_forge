from __future__ import annotations
import decimal
import random
import re
from . import base as oracle
from .base import OracleCompiler
from .base import OracleDialect
from .base import OracleExecutionContext
from .types import _OracleDateLiteralRender
from ... import exc
from ... import util
from ...engine import cursor as _cursor
from ...engine import interfaces
from ...engine import processors
from ...sql import sqltypes
from ...sql._typing import is_sql_compiler
def convert_cx_oracle_constant(value):
    if isinstance(value, str):
        try:
            int_val = int(value)
        except ValueError:
            value = value.upper()
            return getattr(self.dbapi, value)
        else:
            return int_val
    else:
        return value