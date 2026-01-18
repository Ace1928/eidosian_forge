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
def onconflict_sql(self, expression: exp.OnConflict) -> str:
    conflict = 'ON DUPLICATE KEY' if expression.args.get('duplicate') else 'ON CONFLICT'
    constraint = self.sql(expression, 'constraint')
    constraint = f' ON CONSTRAINT {constraint}' if constraint else ''
    conflict_keys = self.expressions(expression, key='conflict_keys', flat=True)
    conflict_keys = f'({conflict_keys}) ' if conflict_keys else ' '
    action = self.sql(expression, 'action')
    expressions = self.expressions(expression, flat=True)
    if expressions:
        set_keyword = 'SET ' if self.DUPLICATE_KEY_UPDATE_WITH_SET else ''
        expressions = f' {set_keyword}{expressions}'
    return f'{conflict}{constraint}{conflict_keys}{action}{expressions}'