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
def build_logarithm(args: t.List, dialect: Dialect) -> exp.Func:
    this = seq_get(args, 0)
    expression = seq_get(args, 1)
    if expression:
        if not dialect.LOG_BASE_FIRST:
            this, expression = (expression, this)
        return exp.Log(this=this, expression=expression)
    return (exp.Ln if dialect.parser_class.LOG_DEFAULTS_TO_LN else exp.Log)(this=this)