from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def rle_get_at(rle, pos: int):
    """
    Return the attribute at offset pos.
    """
    x = 0
    if pos < 0:
        return None
    for a, run in rle:
        if x + run > pos:
            return a
        x += run
    return None