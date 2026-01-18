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
def shard_body_tail(num_rows: int, sbody):
    """
    Return a new shard tail that follows this shard body.
    """
    shard_tail = []
    col_gap = 0
    for done_rows, content_iter, cview in sbody:
        cols, rows = cview[2:4]
        done_rows += num_rows
        if done_rows == rows:
            col_gap += cols
            continue
        shard_tail.append((col_gap, done_rows, content_iter, cview))
        col_gap = 0
    return shard_tail