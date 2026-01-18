from __future__ import annotations
import logging
import typing as t
from enum import Enum, auto
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.helper import AutoName, flatten, is_int, seq_get
from sqlglot.jsonpath import parse as parse_json_path
from sqlglot.parser import Parser
from sqlglot.time import TIMEZONES, format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import new_trie
def filter_array_using_unnest(self: Generator, expression: exp.ArrayFilter) -> str:
    cond = expression.expression
    if isinstance(cond, exp.Lambda) and len(cond.expressions) == 1:
        alias = cond.expressions[0]
        cond = cond.this
    elif isinstance(cond, exp.Predicate):
        alias = '_u'
    else:
        self.unsupported('Unsupported filter condition')
        return ''
    unnest = exp.Unnest(expressions=[expression.this])
    filtered = exp.select(alias).from_(exp.alias_(unnest, None, table=[alias])).where(cond)
    return self.sql(exp.Array(expressions=[filtered]))