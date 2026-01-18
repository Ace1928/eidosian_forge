from __future__ import annotations
import contextlib
import logging
import math
import os
import pathlib
import re
import sys
import tempfile
from functools import partial
from hashlib import md5
from importlib.metadata import version
from typing import (
from urllib.parse import urlsplit
def merge_offset_ranges(paths: list[str], starts: list[int] | int, ends: list[int] | int, max_gap: int=0, max_block: int | None=None, sort: bool=True) -> tuple[list[str], list[int], list[int]]:
    """Merge adjacent byte-offset ranges when the inter-range
    gap is <= `max_gap`, and when the merged byte range does not
    exceed `max_block` (if specified). By default, this function
    will re-order the input paths and byte ranges to ensure sorted
    order. If the user can guarantee that the inputs are already
    sorted, passing `sort=False` will skip the re-ordering.
    """
    if not isinstance(paths, list):
        raise TypeError
    if not isinstance(starts, list):
        starts = [starts] * len(paths)
    if not isinstance(ends, list):
        ends = [ends] * len(paths)
    if len(starts) != len(paths) or len(ends) != len(paths):
        raise ValueError
    if len(starts) <= 1:
        return (paths, starts, ends)
    starts = [s or 0 for s in starts]
    if sort:
        paths, starts, ends = (list(v) for v in zip(*sorted(zip(paths, starts, ends))))
    if paths:
        new_paths = paths[:1]
        new_starts = starts[:1]
        new_ends = ends[:1]
        for i in range(1, len(paths)):
            if paths[i] == paths[i - 1] and new_ends[-1] is None:
                continue
            elif paths[i] != paths[i - 1] or starts[i] - new_ends[-1] > max_gap or (max_block is not None and ends[i] - new_starts[-1] > max_block):
                new_paths.append(paths[i])
                new_starts.append(starts[i])
                new_ends.append(ends[i])
            else:
                new_ends[-1] = ends[i]
        return (new_paths, new_starts, new_ends)
    return (paths, starts, ends)