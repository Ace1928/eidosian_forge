from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def offset_sql(self, expression: exp.Offset) -> str:
    return f'{super().offset_sql(expression)} ROWS'