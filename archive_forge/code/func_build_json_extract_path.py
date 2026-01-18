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
def build_json_extract_path(expr_type: t.Type[F], zero_based_indexing: bool=True, arrow_req_json_type: bool=False) -> t.Callable[[t.List], F]:

    def _builder(args: t.List) -> F:
        segments: t.List[exp.JSONPathPart] = [exp.JSONPathRoot()]
        for arg in args[1:]:
            if not isinstance(arg, exp.Literal):
                return expr_type.from_arg_list(args)
            text = arg.name
            if is_int(text):
                index = int(text)
                segments.append(exp.JSONPathSubscript(this=index if zero_based_indexing else index - 1))
            else:
                segments.append(exp.JSONPathKey(this=text))
        del args[2:]
        return expr_type(this=seq_get(args, 0), expression=exp.JSONPath(expressions=segments), only_json_types=arrow_req_json_type)
    return _builder