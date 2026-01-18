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
def mergetreettl_sql(self, expression: exp.MergeTreeTTL) -> str:
    where = self.sql(expression, 'where')
    group = self.sql(expression, 'group')
    aggregates = self.expressions(expression, key='aggregates')
    aggregates = self.seg('SET') + self.seg(aggregates) if aggregates else ''
    if not (where or group or aggregates) and len(expression.expressions) == 1:
        return f'TTL {self.expressions(expression, flat=True)}'
    return f'TTL{self.seg(self.expressions(expression))}{where}{group}{aggregates}'