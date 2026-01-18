from __future__ import annotations
import logging
import typing as t
from collections import defaultdict
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie
def build_extract_json_with_path(expr_type: t.Type[E]) -> t.Callable[[t.List, Dialect], E]:

    def _builder(args: t.List, dialect: Dialect) -> E:
        expression = expr_type(this=seq_get(args, 0), expression=dialect.to_json_path(seq_get(args, 1)))
        if len(args) > 2 and expr_type is exp.JSONExtract:
            expression.set('expressions', args[2:])
        return expression
    return _builder