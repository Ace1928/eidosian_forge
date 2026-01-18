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
def fetchall_for_returning(self, cursor, *, _internal=False):
    compiled = self.compiled
    if not _internal and compiled is None or not is_sql_compiler(compiled) or (not compiled._oracle_returning):
        raise NotImplementedError('execution context was not prepared for Oracle RETURNING')
    numcols = len(self.out_parameters)
    return list(zip(*[[val for stmt_result in self.out_parameters[f'ret_{j}'].values for val in stmt_result or ()] for j in range(numcols)]))