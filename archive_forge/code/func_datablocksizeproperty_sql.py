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
def datablocksizeproperty_sql(self, expression: exp.DataBlocksizeProperty) -> str:
    default = expression.args.get('default')
    minimum = expression.args.get('minimum')
    maximum = expression.args.get('maximum')
    if default or minimum or maximum:
        if default:
            prop = 'DEFAULT'
        elif minimum:
            prop = 'MINIMUM'
        else:
            prop = 'MAXIMUM'
        return f'{prop} DATABLOCKSIZE'
    units = expression.args.get('units')
    units = f' {units}' if units else ''
    return f'DATABLOCKSIZE={self.sql(expression, 'size')}{units}'