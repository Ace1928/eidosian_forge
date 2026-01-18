from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def normalize_identifier(self, expression: E) -> E:
    if isinstance(expression, exp.Identifier):
        parent = expression.parent
        while isinstance(parent, exp.Dot):
            parent = parent.parent
        case_sensitive = isinstance(parent, exp.UserDefinedFunction) or (isinstance(parent, exp.Table) and parent.db and (parent.meta.get('quoted_table') or not parent.meta.get('maybe_column'))) or expression.meta.get('is_table')
        if not case_sensitive:
            expression.set('this', expression.this.lower())
    return expression