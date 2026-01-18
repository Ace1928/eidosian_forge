from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def mod_sql(self, expression: exp.Mod) -> str:
    return self.binary(expression, 'MOD')