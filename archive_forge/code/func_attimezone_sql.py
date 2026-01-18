from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def attimezone_sql(self, expression: exp.AtTimeZone) -> str:
    parent = expression.parent
    if not isinstance(parent, exp.Cast) or not parent.to.is_type('text'):
        return self.func('TIMESTAMP', self.func('DATETIME', expression.this, expression.args.get('zone')))
    return super().attimezone_sql(expression)