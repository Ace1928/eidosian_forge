from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def ceil_week(d: datetime) -> datetime:
    """
    Round up to the start of the next month

    Week start on are on 1st, 8th, 15th and 22nd
    day of the month
    """
    _d_floor = floor_week(d)
    if d == _d_floor:
        return d
    if d.day >= 22:
        return d.min.replace(d.year, d.month, d.day) + ONE_WEEK
    return _d_floor + ONE_WEEK