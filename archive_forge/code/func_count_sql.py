from __future__ import annotations
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import Dialect, rename_func
def count_sql(self, expression: exp.Count) -> str:
    this = expression.this
    if isinstance(this, exp.Distinct):
        return self.func('COUNTD', *this.expressions)
    return self.func('COUNT', this)