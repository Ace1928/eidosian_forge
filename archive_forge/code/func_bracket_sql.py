from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.dialects.dialect import rename_func, unit_to_var
from sqlglot.dialects.hive import _build_with_ignore_nulls
from sqlglot.dialects.spark2 import Spark2, temporary_storage_provider
from sqlglot.helper import ensure_list, seq_get
from sqlglot.transforms import (
def bracket_sql(self, expression: exp.Bracket) -> str:
    if expression.args.get('safe'):
        key = seq_get(self.bracket_offset_expressions(expression), 0)
        return self.func('TRY_ELEMENT_AT', expression.this, key)
    return super().bracket_sql(expression)