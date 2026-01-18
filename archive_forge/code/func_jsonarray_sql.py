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
def jsonarray_sql(self, expression: exp.JSONArray) -> str:
    null_handling = expression.args.get('null_handling')
    null_handling = f' {null_handling}' if null_handling else ''
    return_type = self.sql(expression, 'return_type')
    return_type = f' RETURNING {return_type}' if return_type else ''
    strict = ' STRICT' if expression.args.get('strict') else ''
    return self.func('JSON_ARRAY', *expression.expressions, suffix=f'{null_handling}{return_type}{strict})')