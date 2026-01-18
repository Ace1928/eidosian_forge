from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def split_num_words(value: str, sep: str, min_num_words: int, fill_from_start: bool=True) -> t.List[t.Optional[str]]:
    """
    Perform a split on a value and return N words as a result with `None` used for words that don't exist.

    Args:
        value: The value to be split.
        sep: The value to use to split on.
        min_num_words: The minimum number of words that are going to be in the result.
        fill_from_start: Indicates that if `None` values should be inserted at the start or end of the list.

    Examples:
        >>> split_num_words("db.table", ".", 3)
        [None, 'db', 'table']
        >>> split_num_words("db.table", ".", 3, fill_from_start=False)
        ['db', 'table', None]
        >>> split_num_words("db.table", ".", 1)
        ['db', 'table']

    Returns:
        The list of words returned by `split`, possibly augmented by a number of `None` values.
    """
    words = value.split(sep)
    if fill_from_start:
        return [None] * (min_num_words - len(words)) + words
    return words + [None] * (min_num_words - len(words))