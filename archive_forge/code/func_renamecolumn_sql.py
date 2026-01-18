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
def renamecolumn_sql(self, expression: exp.RenameColumn) -> str:
    exists = ' IF EXISTS' if expression.args.get('exists') else ''
    old_column = self.sql(expression, 'this')
    new_column = self.sql(expression, 'to')
    return f'RENAME COLUMN{exists} {old_column} TO {new_column}'