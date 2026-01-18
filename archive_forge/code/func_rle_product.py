from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def rle_product(rle1, rle2):
    """
    Merge the runs of rle1 and rle2 like this:
    eg.
    rle1 = [ ("a", 10), ("b", 5) ]
    rle2 = [ ("Q", 5), ("P", 10) ]
    rle_product: [ (("a","Q"), 5), (("a","P"), 5), (("b","P"), 5) ]

    rle1 and rle2 are assumed to cover the same total run.
    """
    i1 = i2 = 1
    if not rle1 or not rle2:
        return []
    a1, r1 = rle1[0]
    a2, r2 = rle2[0]
    result = []
    while r1 and r2:
        r = min(r1, r2)
        rle_append_modify(result, ((a1, a2), r))
        r1 -= r
        if r1 == 0 and i1 < len(rle1):
            a1, r1 = rle1[i1]
            i1 += 1
        r2 -= r
        if r2 == 0 and i2 < len(rle2):
            a2, r2 = rle2[i2]
            i2 += 1
    return result