from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def date_add_sql(kind: str) -> t.Callable[[generator.Generator, exp.Expression], str]:

    def func(self: generator.Generator, expression: exp.Expression) -> str:
        return self.func(f'DATE_{kind}', expression.this, exp.Interval(this=expression.expression, unit=unit_to_var(expression)))
    return func