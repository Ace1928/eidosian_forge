from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def interval_sql(self, expression: exp.Interval) -> str:
    if expression.this and expression.text('unit').upper().startswith('WEEK'):
        return f"({expression.this.name} * INTERVAL '7' DAY)"
    return super().interval_sql(expression)