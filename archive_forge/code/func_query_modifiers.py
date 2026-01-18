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
def query_modifiers(self, expression: exp.Expression, *sqls: str) -> str:
    limit = expression.args.get('limit')
    if self.LIMIT_FETCH == 'LIMIT' and isinstance(limit, exp.Fetch):
        limit = exp.Limit(expression=exp.maybe_copy(limit.args.get('count')))
    elif self.LIMIT_FETCH == 'FETCH' and isinstance(limit, exp.Limit):
        limit = exp.Fetch(direction='FIRST', count=exp.maybe_copy(limit.expression))
    options = self.expressions(expression, key='options')
    if options:
        options = f' OPTION{self.wrap(options)}'
    return csv(*sqls, *[self.sql(join) for join in expression.args.get('joins') or []], self.sql(expression, 'connect'), self.sql(expression, 'match'), *[self.sql(lateral) for lateral in expression.args.get('laterals') or []], self.sql(expression, 'prewhere'), self.sql(expression, 'where'), self.sql(expression, 'group'), self.sql(expression, 'having'), *[gen(self, expression) for gen in self.AFTER_HAVING_MODIFIER_TRANSFORMS.values()], self.sql(expression, 'order'), *self.offset_limit_modifiers(expression, isinstance(limit, exp.Fetch), limit), *self.after_limit_modifiers(expression), options, sep='')