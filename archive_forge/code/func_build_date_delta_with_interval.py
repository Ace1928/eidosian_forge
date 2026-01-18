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
def build_date_delta_with_interval(expression_class: t.Type[E]) -> t.Callable[[t.List], t.Optional[E]]:

    def _builder(args: t.List) -> t.Optional[E]:
        if len(args) < 2:
            return None
        interval = args[1]
        if not isinstance(interval, exp.Interval):
            raise ParseError(f"INTERVAL expression expected but got '{interval}'")
        expression = interval.this
        if expression and expression.is_string:
            expression = exp.Literal.number(expression.this)
        return expression_class(this=args[0], expression=expression, unit=unit_to_str(interval))
    return _builder