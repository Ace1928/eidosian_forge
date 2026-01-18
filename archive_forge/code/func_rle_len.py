from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def rle_len(rle: Iterable[tuple[typing.Any, int]]) -> int:
    """
    Return the number of characters covered by a run length
    encoded attribute list.
    """
    run = 0
    for v in rle:
        if not isinstance(v, tuple):
            raise TypeError(rle)
        _a, r = v
        run += r
    return run