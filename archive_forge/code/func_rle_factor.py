from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def rle_factor(rle):
    """
    Inverse of rle_product.
    """
    rle1 = []
    rle2 = []
    for (a1, a2), r in rle:
        rle_append_modify(rle1, (a1, r))
        rle_append_modify(rle2, (a2, r))
    return (rle1, rle2)