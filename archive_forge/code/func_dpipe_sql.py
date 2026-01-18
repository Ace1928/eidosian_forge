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
def dpipe_sql(self, expression: exp.DPipe) -> str:
    if self.dialect.STRICT_STRING_CONCAT and expression.args.get('safe'):
        return self.func('CONCAT', *(exp.cast(e, exp.DataType.Type.TEXT) for e in expression.flatten()))
    return self.binary(expression, '||')