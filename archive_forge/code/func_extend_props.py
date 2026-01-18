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
def extend_props(temp_props: t.Optional[exp.Properties]) -> None:
    nonlocal properties
    if properties and temp_props:
        properties.expressions.extend(temp_props.expressions)
    elif temp_props:
        properties = temp_props