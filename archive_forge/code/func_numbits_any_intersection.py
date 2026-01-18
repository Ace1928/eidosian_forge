from __future__ import annotations
import json
import sqlite3
from itertools import zip_longest
from typing import Iterable
def numbits_any_intersection(numbits1: bytes, numbits2: bytes) -> bool:
    """Is there any number that appears in both numbits?

    Determine whether two number sets have a non-empty intersection. This is
    faster than computing the intersection.

    Returns:
        A bool, True if there is any number in both `numbits1` and `numbits2`.
    """
    byte_pairs = zip_longest(numbits1, numbits2, fillvalue=0)
    return any((b1 & b2 for b1, b2 in byte_pairs))