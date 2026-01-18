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
def can_identify(self, text: str, identify: str | bool='safe') -> bool:
    """Checks if text can be identified given an identify option.

        Args:
            text: The text to check.
            identify:
                `"always"` or `True`: Always returns `True`.
                `"safe"`: Only returns `True` if the identifier is case-insensitive.

        Returns:
            Whether the given text can be identified.
        """
    if identify is True or identify == 'always':
        return True
    if identify == 'safe':
        return not self.case_sensitive(text)
    return False