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
def naked_property(self, expression: exp.Property) -> str:
    property_name = exp.Properties.PROPERTY_TO_NAME.get(expression.__class__)
    if not property_name:
        self.unsupported(f'Unsupported property {expression.__class__.__name__}')
    return f'{property_name} {self.sql(expression, 'this')}'