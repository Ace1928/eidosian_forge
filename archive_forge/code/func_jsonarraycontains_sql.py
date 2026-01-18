from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def jsonarraycontains_sql(self, expression: exp.JSONArrayContains) -> str:
    return f'{self.sql(expression, 'this')} MEMBER OF({self.sql(expression, 'expression')})'