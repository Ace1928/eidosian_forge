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
def datatypeparam_sql(self, expression: exp.DataTypeParam) -> str:
    this = self.sql(expression, 'this')
    specifier = self.sql(expression, 'expression')
    specifier = f' {specifier}' if specifier and self.DATA_TYPE_SPECIFIERS_ALLOWED else ''
    return f'{this}{specifier}'