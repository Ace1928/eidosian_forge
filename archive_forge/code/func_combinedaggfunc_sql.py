from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def combinedaggfunc_sql(self, expression: exp.CombinedAggFunc) -> str:
    return self.anonymousaggfunc_sql(expression)