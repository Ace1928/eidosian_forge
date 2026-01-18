from __future__ import annotations
import json
import sqlite3
from itertools import zip_longest
from typing import Iterable
def nums_to_numbits(nums: Iterable[int]) -> bytes:
    """Convert `nums` into a numbits.

    Arguments:
        nums: a reusable iterable of integers, the line numbers to store.

    Returns:
        A binary blob.
    """
    try:
        nbytes = max(nums) // 8 + 1
    except ValueError:
        return b''
    b = bytearray(nbytes)
    for num in nums:
        b[num // 8] |= 1 << num % 8
    return bytes(b)