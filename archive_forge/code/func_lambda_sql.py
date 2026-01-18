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
def lambda_sql(self, expression: exp.Lambda, arrow_sep: str='->') -> str:
    args = self.expressions(expression, flat=True)
    args = f'({args})' if len(args.split(',')) > 1 else args
    return f'{args} {arrow_sep} {self.sql(expression, 'this')}'