import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
def joinstem(cutpoint: Optional[int]=0, words: Optional[Iterable[str]]=None) -> str:
    """
    Join stem of each word in words into a string for regex.

    Each word is truncated at cutpoint.

    Cutpoint is usually negative indicating the number of letters to remove
    from the end of each word.

    >>> joinstem(-2, ["ephemeris", "iris", ".*itis"])
    '(?:ephemer|ir|.*it)'

    >>> joinstem(None, ["ephemeris"])
    '(?:ephemeris)'

    >>> joinstem(5, None)
    '(?:)'
    """
    return enclose('|'.join((w[:cutpoint] for w in words or [])))