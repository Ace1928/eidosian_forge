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
def comprehension_sql(self, expression: exp.Comprehension) -> str:
    this = self.sql(expression, 'this')
    expr = self.sql(expression, 'expression')
    iterator = self.sql(expression, 'iterator')
    condition = self.sql(expression, 'condition')
    condition = f' IF {condition}' if condition else ''
    return f'{this} FOR {expr} IN {iterator}{condition}'