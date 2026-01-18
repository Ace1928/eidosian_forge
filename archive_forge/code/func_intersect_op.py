from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def intersect_op(self, expression: exp.Intersect) -> str:
    if not expression.args.get('distinct'):
        self.unsupported('INTERSECT without DISTINCT is not supported in BigQuery')
    return f'INTERSECT{(' DISTINCT' if expression.args.get('distinct') else ' ALL')}'