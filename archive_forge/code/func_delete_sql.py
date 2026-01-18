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
def delete_sql(self, expression: exp.Delete) -> str:
    this = self.sql(expression, 'this')
    this = f' FROM {this}' if this else ''
    using = self.sql(expression, 'using')
    using = f' USING {using}' if using else ''
    where = self.sql(expression, 'where')
    returning = self.sql(expression, 'returning')
    limit = self.sql(expression, 'limit')
    tables = self.expressions(expression, key='tables')
    tables = f' {tables}' if tables else ''
    if self.RETURNING_END:
        expression_sql = f'{this}{using}{where}{returning}{limit}'
    else:
        expression_sql = f'{returning}{this}{using}{where}{limit}'
    return self.prepend_ctes(expression, f'DELETE{tables}{expression_sql}')