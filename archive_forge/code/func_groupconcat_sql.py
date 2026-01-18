from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def groupconcat_sql(self, expression: exp.GroupConcat) -> str:
    this = expression.this
    distinct = expression.find(exp.Distinct)
    if distinct:
        this = distinct.expressions[0]
        distinct_sql = 'DISTINCT '
    else:
        distinct_sql = ''
    if isinstance(expression.this, exp.Order):
        self.unsupported("SQLite GROUP_CONCAT doesn't support ORDER BY.")
        if expression.this.this and (not distinct):
            this = expression.this.this
    separator = expression.args.get('separator')
    return f'GROUP_CONCAT({distinct_sql}{self.format_args(this, separator)})'