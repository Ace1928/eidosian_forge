from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def shift_limits_down(candidate_limits: TupleInt2, original_limits: TupleInt2, width: int) -> TupleInt2:
    """
    Shift candidate limits down so that they can be a multiple of width

    If the shift would exclude any of the original_limits (high),
    candidate limits are returned.

    The goal of this function is to convert abitrary limits into "nicer"
    ones.
    """
    low, high = candidate_limits
    low_orig, high_orig = original_limits
    l, m = divmod(low, width)
    if isclose_abs(m / width, 1):
        l += 1
    low_new = l * width
    diff = low - low_new
    high_new = high - diff
    if high_orig <= high_new:
        return (low_new, high_new)
    return candidate_limits