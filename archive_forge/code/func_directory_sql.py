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
def directory_sql(self, expression: exp.Directory) -> str:
    local = 'LOCAL ' if expression.args.get('local') else ''
    row_format = self.sql(expression, 'row_format')
    row_format = f' {row_format}' if row_format else ''
    return f'{local}DIRECTORY {self.sql(expression, 'this')}{row_format}'