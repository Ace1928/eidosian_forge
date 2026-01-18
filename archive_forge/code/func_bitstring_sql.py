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
def bitstring_sql(self, expression: exp.BitString) -> str:
    this = self.sql(expression, 'this')
    if self.dialect.BIT_START:
        return f'{self.dialect.BIT_START}{this}{self.dialect.BIT_END}'
    return f'{int(this, 2)}'