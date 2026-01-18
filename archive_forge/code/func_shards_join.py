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
def shards_join(shard_lists):
    """
    Return the result of joining shard lists horizontally.
    All shards lists must have the same number of rows.
    """
    shards_iters = [iter(sl) for sl in shard_lists]
    shards_current = [next(i) for i in shards_iters]
    new_shards = []
    while True:
        new_cviews = []
        num_rows = min((r for r, cv in shards_current))
        shards_next = []
        for rows, cviews in shards_current:
            if cviews:
                new_cviews.extend(cviews)
            shards_next.append((rows - num_rows, None))
        shards_current = shards_next
        new_shards.append((num_rows, new_cviews))
        try:
            for i in range(len(shards_current)):
                if shards_current[i][0] > 0:
                    continue
                shards_current[i] = next(shards_iters[i])
        except StopIteration:
            break
    return new_shards