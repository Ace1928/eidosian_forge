from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.dialects.dialect import rename_func, unit_to_var
from sqlglot.dialects.hive import _build_with_ignore_nulls
from sqlglot.dialects.spark2 import Spark2, temporary_storage_provider
from sqlglot.helper import ensure_list, seq_get
from sqlglot.transforms import (
def datediff_sql(self, expression: exp.DateDiff) -> str:
    end = self.sql(expression, 'this')
    start = self.sql(expression, 'expression')
    if expression.unit:
        return self.func('DATEDIFF', unit_to_var(expression), start, end)
    return self.func('DATEDIFF', end, start)