from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def currenttimestamp_sql(self, expression: exp.CurrentTimestamp) -> str:
    this = expression.this
    return self.func('CURRENT_TIMESTAMP', this) if this else 'CURRENT_TIMESTAMP'