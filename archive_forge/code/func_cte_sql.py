from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def cte_sql(self, expression: exp.CTE) -> str:
    if expression.args.get('scalar'):
        this = self.sql(expression, 'this')
        alias = self.sql(expression, 'alias')
        return f'{this} AS {alias}'
    return super().cte_sql(expression)