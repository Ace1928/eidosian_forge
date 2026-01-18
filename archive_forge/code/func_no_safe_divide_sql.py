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
def no_safe_divide_sql(self: Generator, expression: exp.SafeDivide) -> str:
    n = self.sql(expression, 'this')
    d = self.sql(expression, 'expression')
    return f'IF(({d}) <> 0, ({n}) / ({d}), NULL)'