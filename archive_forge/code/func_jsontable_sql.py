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
def jsontable_sql(self, expression: exp.JSONTable) -> str:
    this = self.sql(expression, 'this')
    path = self.sql(expression, 'path')
    path = f', {path}' if path else ''
    error_handling = expression.args.get('error_handling')
    error_handling = f' {error_handling}' if error_handling else ''
    empty_handling = expression.args.get('empty_handling')
    empty_handling = f' {empty_handling}' if empty_handling else ''
    schema = self.sql(expression, 'schema')
    return self.func('JSON_TABLE', this, suffix=f'{path}{error_handling}{empty_handling} {schema})')