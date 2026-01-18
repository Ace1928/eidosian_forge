from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def select_sql(self, expression: exp.Select) -> str:
    if expression.args.get('offset'):
        if not expression.args.get('order'):
            expression.order_by(exp.select(exp.null()).subquery(), copy=False)
        limit = expression.args.get('limit')
        if isinstance(limit, exp.Limit):
            limit.replace(exp.Fetch(direction='FIRST', count=limit.expression))
    return super().select_sql(expression)