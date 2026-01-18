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
def renametable_sql(self, expression: exp.RenameTable) -> str:
    if not self.RENAME_TABLE_WITH_DB:
        expression = expression.transform(lambda n: exp.table_(n.this) if isinstance(n, exp.Table) else n).assert_is(exp.RenameTable)
    this = self.sql(expression, 'this')
    return f'RENAME TO {this}'