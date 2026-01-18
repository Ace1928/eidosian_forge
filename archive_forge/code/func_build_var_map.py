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
def build_var_map(args: t.List) -> exp.StarMap | exp.VarMap:
    if len(args) == 1 and args[0].is_star:
        return exp.StarMap(this=args[0])
    keys = []
    values = []
    for i in range(0, len(args), 2):
        keys.append(args[i])
        values.append(args[i + 1])
    return exp.VarMap(keys=exp.array(*keys, copy=False), values=exp.array(*values, copy=False))