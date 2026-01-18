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
def inline_array_unless_query(self: Generator, expression: exp.Array) -> str:
    elem = seq_get(expression.expressions, 0)
    if isinstance(elem, exp.Expression) and elem.find(exp.Query):
        return self.func('ARRAY', elem)
    return inline_array_sql(self, expression)