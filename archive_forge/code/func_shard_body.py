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
def shard_body(cviews, shard_tail, create_iter: bool=True, iter_default=None):
    """
    Return a list of (done_rows, content_iter, cview) tuples for
    this shard and shard tail.

    If a canvas in cviews is None (eg. when unchanged from
    shard_cviews_delta()) or if create_iter is False then no
    iterator is created for content_iter.

    iter_default is the value used for content_iter when no iterator
    is created.
    """
    col = 0
    body = []
    cviews_iter = iter(cviews)
    for col_gap, done_rows, content_iter, tail_cview in shard_tail:
        while col_gap:
            try:
                cview = next(cviews_iter)
            except StopIteration:
                break
            trim_left, trim_top, cols, rows, attr_map, canv = cview[:6]
            col += cols
            col_gap -= cols
            if col_gap < 0:
                raise CanvasError('cviews overflow gaps in shard_tail!')
            if create_iter and canv:
                new_iter = canv.content(trim_left, trim_top, cols, rows, attr_map)
            else:
                new_iter = iter_default
            body.append((0, new_iter, cview))
        body.append((done_rows, content_iter, tail_cview))
    for cview in cviews_iter:
        trim_left, trim_top, cols, rows, attr_map, canv = cview[:6]
        if create_iter and canv:
            new_iter = canv.content(trim_left, trim_top, cols, rows, attr_map)
        else:
            new_iter = iter_default
        body.append((0, new_iter, cview))
    return body