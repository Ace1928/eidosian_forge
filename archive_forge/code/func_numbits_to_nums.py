from __future__ import annotations
import json
import sqlite3
from itertools import zip_longest
from typing import Iterable
def numbits_to_nums(numbits: bytes) -> list[int]:
    """Convert a numbits into a list of numbers.

    Arguments:
        numbits: a binary blob, the packed number set.

    Returns:
        A list of ints.

    When registered as a SQLite function by :func:`register_sqlite_functions`,
    this returns a string, a JSON-encoded list of ints.

    """
    nums = []
    for byte_i, byte in enumerate(numbits):
        for bit_i in range(8):
            if byte & 1 << bit_i:
                nums.append(byte_i * 8 + bit_i)
    return nums