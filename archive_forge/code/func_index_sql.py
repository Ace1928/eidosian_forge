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
def index_sql(self, expression: exp.Index) -> str:
    unique = 'UNIQUE ' if expression.args.get('unique') else ''
    primary = 'PRIMARY ' if expression.args.get('primary') else ''
    amp = 'AMP ' if expression.args.get('amp') else ''
    name = self.sql(expression, 'this')
    name = f'{name} ' if name else ''
    table = self.sql(expression, 'table')
    table = f'{self.INDEX_ON} {table}' if table else ''
    index = 'INDEX ' if not table else ''
    params = self.sql(expression, 'params')
    return f'{unique}{primary}{amp}{index}{name}{table}{params}'