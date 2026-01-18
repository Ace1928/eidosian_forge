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
def order_sql(self, expression: exp.Order, flat: bool=False) -> str:
    this = self.sql(expression, 'this')
    this = f'{this} ' if this else this
    siblings = 'SIBLINGS ' if expression.args.get('siblings') else ''
    order = self.op_expressions(f'{this}ORDER {siblings}BY', expression, flat=this or flat)
    interpolated_values = [f'{self.sql(named_expression, 'alias')} AS {self.sql(named_expression, 'this')}' for named_expression in expression.args.get('interpolate') or []]
    interpolate = f' INTERPOLATE ({', '.join(interpolated_values)})' if interpolated_values else ''
    return f'{order}{interpolate}'