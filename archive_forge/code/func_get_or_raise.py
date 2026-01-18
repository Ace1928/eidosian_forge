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
@classmethod
def get_or_raise(cls, dialect: DialectType) -> Dialect:
    """
        Look up a dialect in the global dialect registry and return it if it exists.

        Args:
            dialect: The target dialect. If this is a string, it can be optionally followed by
                additional key-value pairs that are separated by commas and are used to specify
                dialect settings, such as whether the dialect's identifiers are case-sensitive.

        Example:
            >>> dialect = dialect_class = get_or_raise("duckdb")
            >>> dialect = get_or_raise("mysql, normalization_strategy = case_sensitive")

        Returns:
            The corresponding Dialect instance.
        """
    if not dialect:
        return cls()
    if isinstance(dialect, _Dialect):
        return dialect()
    if isinstance(dialect, Dialect):
        return dialect
    if isinstance(dialect, str):
        try:
            dialect_name, *kv_pairs = dialect.split(',')
            kwargs = {k.strip(): v.strip() for k, v in (kv.split('=') for kv in kv_pairs)}
        except ValueError:
            raise ValueError(f"Invalid dialect format: '{dialect}'. Please use the correct format: 'dialect [, k1 = v2 [, ...]]'.")
        result = cls.get(dialect_name.strip())
        if not result:
            from difflib import get_close_matches
            similar = seq_get(get_close_matches(dialect_name, cls.classes, n=1), 0) or ''
            if similar:
                similar = f' Did you mean {similar}?'
            raise ValueError(f"Unknown dialect '{dialect_name}'.{similar}")
        return result(**kwargs)
    raise ValueError(f"Invalid dialect type for '{dialect}': '{type(dialect)}'.")