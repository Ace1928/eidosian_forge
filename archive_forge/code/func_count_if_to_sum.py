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
def count_if_to_sum(self: Generator, expression: exp.CountIf) -> str:
    cond = expression.this
    if isinstance(expression.this, exp.Distinct):
        cond = expression.this.expressions[0]
        self.unsupported('DISTINCT is not supported when converting COUNT_IF to SUM')
    return self.func('sum', exp.func('if', cond, 1, 0))