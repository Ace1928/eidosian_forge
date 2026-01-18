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
def op_expressions(self, op: str, expression: exp.Expression, flat: bool=False) -> str:
    flat = flat or isinstance(expression.parent, exp.Properties)
    expressions_sql = self.expressions(expression, flat=flat)
    if flat:
        return f'{op} {expressions_sql}'
    return f'{self.seg(op)}{(self.sep() if expressions_sql else '')}{expressions_sql}'