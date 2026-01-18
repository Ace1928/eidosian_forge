from __future__ import annotations
import json
import sqlite3
from itertools import zip_longest
from typing import Iterable
def numbits_intersection(numbits1: bytes, numbits2: bytes) -> bytes:
    """Compute the intersection of two numbits.

    Returns:
        A new numbits, the intersection `numbits1` and `numbits2`.
    """
    byte_pairs = zip_longest(numbits1, numbits2, fillvalue=0)
    intersection_bytes = bytes((b1 & b2 for b1, b2 in byte_pairs))
    return intersection_bytes.rstrip(b'\x00')