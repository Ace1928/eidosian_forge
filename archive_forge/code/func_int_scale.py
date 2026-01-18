from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def int_scale(val: int, val_range: int, out_range: int) -> int:
    """
    Scale val in the range [0, val_range-1] to an integer in the range
    [0, out_range-1].  This implementation uses the "round-half-up" rounding
    method.

    >>> "%x" % int_scale(0x7, 0x10, 0x10000)
    '7777'
    >>> "%x" % int_scale(0x5f, 0x100, 0x10)
    '6'
    >>> int_scale(2, 6, 101)
    40
    >>> int_scale(1, 3, 4)
    2
    """
    num = int(val * (out_range - 1) * 2 + (val_range - 1))
    dem = (val_range - 1) * 2
    return num // dem