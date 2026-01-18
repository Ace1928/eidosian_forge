from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def schema_sql(self, expression: exp.Schema) -> str:
    for ordered in expression.find_all(exp.Ordered):
        if ordered.args.get('desc') is False:
            ordered.set('desc', None)
    return super().schema_sql(expression)