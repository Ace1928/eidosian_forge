from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def altercolumn_sql(self, expression: exp.AlterColumn) -> str:
    dtype = self.sql(expression, 'dtype')
    if not dtype:
        return super().altercolumn_sql(expression)
    this = self.sql(expression, 'this')
    return f'MODIFY COLUMN {this} {dtype}'