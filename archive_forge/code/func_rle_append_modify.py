from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def rle_append_modify(rle, a_r) -> None:
    """
    Append (a, r) (unpacked from *a_r*) to the rle list rle.
    Merge with last run when possible.

    MODIFIES rle parameter contents. Returns None.
    """
    a, r = a_r
    if not rle or rle[-1][0] != a:
        rle.append((a, r))
        return
    _la, lr = rle[-1]
    rle[-1] = (a, lr + r)