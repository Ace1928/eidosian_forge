from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def returning_sql(self, expression: exp.Returning) -> str:
    into = self.sql(expression, 'into')
    into = self.seg(f'INTO {into}') if into else ''
    return f'{self.seg('OUTPUT')} {self.expressions(expression, flat=True)}{into}'