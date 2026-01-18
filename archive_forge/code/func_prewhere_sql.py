from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def prewhere_sql(self, expression: exp.PreWhere) -> str:
    this = self.indent(self.sql(expression, 'this'))
    return f'{self.seg('PREWHERE')}{self.sep()}{this}'