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
def concat_ws_to_dpipe_sql(self: Generator, expression: exp.ConcatWs) -> str:
    delim, *rest_args = expression.expressions
    return self.sql(reduce(lambda x, y: exp.DPipe(this=x, expression=exp.DPipe(this=delim, expression=y)), rest_args))