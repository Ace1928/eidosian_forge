from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def convert_sql(self, expression: exp.Convert) -> str:
    name = 'TRY_CONVERT' if expression.args.get('safe') else 'CONVERT'
    return self.func(name, expression.this, expression.expression, expression.args.get('style'))