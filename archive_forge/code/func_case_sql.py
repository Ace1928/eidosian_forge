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
def case_sql(self, expression: exp.Case) -> str:
    this = self.sql(expression, 'this')
    statements = [f'CASE {this}' if this else 'CASE']
    for e in expression.args['ifs']:
        statements.append(f'WHEN {self.sql(e, 'this')}')
        statements.append(f'THEN {self.sql(e, 'true')}')
    default = self.sql(expression, 'default')
    if default:
        statements.append(f'ELSE {default}')
    statements.append('END')
    if self.pretty and self.too_wide(statements):
        return self.indent('\n'.join(statements), skip_first=True, skip_last=True)
    return ' '.join(statements)