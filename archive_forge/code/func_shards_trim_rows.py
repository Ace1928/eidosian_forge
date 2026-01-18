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
def shards_trim_rows(shards, keep_rows: int):
    """
    Return the topmost keep_rows rows from shards.
    """
    if keep_rows < 0:
        raise ValueError(keep_rows)
    new_shards = []
    done_rows = 0
    for num_rows, cviews in shards:
        if done_rows >= keep_rows:
            break
        new_cviews = []
        for cv in cviews:
            if cv[3] + done_rows > keep_rows:
                new_cviews.append(cview_trim_rows(cv, keep_rows - done_rows))
            else:
                new_cviews.append(cv)
        if num_rows + done_rows > keep_rows:
            new_shards.append((keep_rows - done_rows, new_cviews))
        else:
            new_shards.append((num_rows, new_cviews))
        done_rows += num_rows
    return new_shards