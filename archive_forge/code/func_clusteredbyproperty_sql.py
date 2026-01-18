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
def clusteredbyproperty_sql(self, expression: exp.ClusteredByProperty) -> str:
    expressions = self.expressions(expression, key='expressions', flat=True)
    sorted_by = self.expressions(expression, key='sorted_by', flat=True)
    sorted_by = f' SORTED BY ({sorted_by})' if sorted_by else ''
    buckets = self.sql(expression, 'buckets')
    return f'CLUSTERED BY ({expressions}){sorted_by} INTO {buckets} BUCKETS'