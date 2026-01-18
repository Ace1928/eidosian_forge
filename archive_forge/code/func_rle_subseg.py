from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def rle_subseg(rle, start: int, end: int):
    """Return a sub segment of a rle list."""
    sub_segment = []
    x = 0
    for a, run in rle:
        if start:
            if start >= run:
                start -= run
                x += run
                continue
            x += start
            run -= start
            start = 0
        if x >= end:
            break
        if x + run > end:
            run = end - x
        x += run
        sub_segment.append((a, run))
    return sub_segment