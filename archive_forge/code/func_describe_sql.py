from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def describe_sql(self, expression: exp.Describe) -> str:
    kind_value = expression.args.get('kind') or 'TABLE'
    kind = f' {kind_value}' if kind_value else ''
    this = f' {self.sql(expression, 'this')}'
    expressions = self.expressions(expression, flat=True)
    expressions = f' {expressions}' if expressions else ''
    return f'DESCRIBE{kind}{this}{expressions}'