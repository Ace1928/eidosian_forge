from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def extract_sql(self, expression: exp.Extract) -> str:
    this = self.sql(expression, 'this')
    if this.upper() != 'QUARTER':
        return super().extract_sql(expression)
    to_char = exp.func('to_char', expression.expression, exp.Literal.string('Q'))
    return self.sql(exp.cast(to_char, exp.DataType.Type.INT))