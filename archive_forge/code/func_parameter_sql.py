from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def parameter_sql(self, expression: exp.Parameter) -> str:
    this = self.sql(expression, 'this')
    expression_sql = self.sql(expression, 'expression')
    parent = expression.parent
    this = f'{this}:{expression_sql}' if expression_sql else this
    if isinstance(parent, exp.EQ) and isinstance(parent.parent, exp.SetItem):
        return this
    return f'${{{this}}}'