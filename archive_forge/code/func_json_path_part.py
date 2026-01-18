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
def json_path_part(self, expression: int | str | exp.JSONPathPart) -> str:
    if isinstance(expression, exp.JSONPathPart):
        transform = self.TRANSFORMS.get(expression.__class__)
        if not callable(transform):
            self.unsupported(f'Unsupported JSONPathPart type {expression.__class__.__name__}')
            return ''
        return transform(self, expression)
    if isinstance(expression, int):
        return str(expression)
    if self.JSON_PATH_SINGLE_QUOTE_ESCAPE:
        escaped = expression.replace("'", "\\'")
        escaped = f"\\'{expression}\\'"
    else:
        escaped = expression.replace('"', '\\"')
        escaped = f'"{escaped}"'
    return escaped