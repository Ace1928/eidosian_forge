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
def partitionboundspec_sql(self, expression: exp.PartitionBoundSpec) -> str:
    if isinstance(expression.this, list):
        return f'IN ({self.expressions(expression, key='this', flat=True)})'
    if expression.this:
        modulus = self.sql(expression, 'this')
        remainder = self.sql(expression, 'expression')
        return f'WITH (MODULUS {modulus}, REMAINDER {remainder})'
    from_expressions = self.expressions(expression, key='from_expressions', flat=True)
    to_expressions = self.expressions(expression, key='to_expressions', flat=True)
    return f'FROM ({from_expressions}) TO ({to_expressions})'