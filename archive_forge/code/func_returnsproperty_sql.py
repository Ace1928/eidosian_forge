from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def returnsproperty_sql(self, expression: exp.ReturnsProperty) -> str:
    table = expression.args.get('table')
    table = f'{table} ' if table else ''
    return f'RETURNS {table}{self.sql(expression, 'this')}'