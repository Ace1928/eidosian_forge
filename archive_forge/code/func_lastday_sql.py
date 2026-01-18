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
def lastday_sql(self, expression: exp.LastDay) -> str:
    if self.LAST_DAY_SUPPORTS_DATE_PART:
        return self.function_fallback_sql(expression)
    unit = expression.text('unit')
    if unit and unit != 'MONTH':
        self.unsupported('Date parts are not supported in LAST_DAY.')
    return self.func('LAST_DAY', expression.this)