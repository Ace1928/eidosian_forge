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
def str_position_sql(self: Generator, expression: exp.StrPosition, generate_instance: bool=False) -> str:
    this = self.sql(expression, 'this')
    substr = self.sql(expression, 'substr')
    position = self.sql(expression, 'position')
    instance = expression.args.get('instance') if generate_instance else None
    position_offset = ''
    if position:
        this = self.func('SUBSTR', this, position)
        position_offset = f' + {position} - 1'
    return self.func('STRPOS', this, substr, instance) + position_offset