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
def bysize(words: Iterable[str]) -> Dict[int, set]:
    """
    From a list of words, return a dict of sets sorted by word length.

    >>> words = ['ant', 'cat', 'dog', 'pig', 'frog', 'goat', 'horse', 'elephant']
    >>> ret = bysize(words)
    >>> sorted(ret[3])
    ['ant', 'cat', 'dog', 'pig']
    >>> ret[5]
    {'horse'}
    """
    res: Dict[int, set] = collections.defaultdict(set)
    for w in words:
        res[len(w)].add(w)
    return res