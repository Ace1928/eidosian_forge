from __future__ import annotations
import bisect as bs
from typing import TYPE_CHECKING
def find_lt(a: list[float], x: float) -> int:
    """Find rightmost value less than x."""
    if (i := bs.bisect_left(a, x)):
        return i - 1
    raise ValueError