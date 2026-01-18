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
def jsoncolumndef_sql(self, expression: exp.JSONColumnDef) -> str:
    path = self.sql(expression, 'path')
    path = f' PATH {path}' if path else ''
    nested_schema = self.sql(expression, 'nested_schema')
    if nested_schema:
        return f'NESTED{path} {nested_schema}'
    this = self.sql(expression, 'this')
    kind = self.sql(expression, 'kind')
    kind = f' {kind}' if kind else ''
    return f'{this}{kind}{path}'