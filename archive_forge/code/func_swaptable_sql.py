from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def swaptable_sql(self, expression: exp.SwapTable) -> str:
    this = self.sql(expression, 'this')
    return f'SWAP WITH {this}'