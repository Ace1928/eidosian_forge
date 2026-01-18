from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
def attrrange(start_offs: int, end_offs: int, destw: int) -> None:
    """
            Add attributes based on attributes between
            start_offs and end_offs.
            """
    if start_offs == end_offs:
        [(at, run)] = arange(start_offs, end_offs)
        rle_append_modify(linea, (at, destw))
        return
    if destw == end_offs - start_offs:
        for at, run in arange(start_offs, end_offs):
            rle_append_modify(linea, (at, run))
        return
    o = start_offs
    for at, run in arange(start_offs, end_offs):
        if o + run == end_offs:
            rle_append_modify(linea, (at, destw))
            return
        tseg = text[o:o + run]
        tseg, cs = apply_target_encoding(tseg)
        segw = rle_len(cs)
        rle_append_modify(linea, (at, segw))
        o += run
        destw -= segw