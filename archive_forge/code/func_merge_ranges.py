from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def merge_ranges(ranges: t.List[t.Tuple[A, A]]) -> t.List[t.Tuple[A, A]]:
    """
    Merges a sequence of ranges, represented as tuples (low, high) whose values
    belong to some totally-ordered set.

    Example:
        >>> merge_ranges([(1, 3), (2, 6)])
        [(1, 6)]
    """
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged