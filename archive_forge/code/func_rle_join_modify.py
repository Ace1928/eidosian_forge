from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def rle_join_modify(rle, rle2) -> None:
    """
    Append attribute list rle2 to rle.
    Merge last run of rle with first run of rle2 when possible.

    MODIFIES attr parameter contents. Returns None.
    """
    if not rle2:
        return
    rle_append_modify(rle, rle2[0])
    rle += rle2[1:]