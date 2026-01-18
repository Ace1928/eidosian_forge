from __future__ import annotations
import logging
import re
import typing as t
from collections import defaultdict
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError, concat_messages
from sqlglot.helper import apply_index_offset, csv, seq_get
from sqlglot.jsonpath import ALL_JSON_PATH_PARTS, JSON_PATH_PART_TRANSFORMS
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def any_sql(self, expression: exp.Any) -> str:
    this = self.sql(expression, 'this')
    if isinstance(expression.this, (*exp.UNWRAPPED_QUERIES, exp.Paren)):
        if isinstance(expression.this, exp.UNWRAPPED_QUERIES):
            this = self.wrap(this)
        return f'ANY{this}'
    return f'ANY {this}'