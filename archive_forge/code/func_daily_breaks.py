from __future__ import annotations
import math
import typing
from datetime import datetime, timedelta, tzinfo
from typing import overload
from zoneinfo import ZoneInfo
import numpy as np
from dateutil.rrule import rrule
from ..utils import get_timezone, isclose_abs
from .date_utils import Interval, align_limits, expand_datetime_limits
from .types import DateFrequency, date_breaks_info
def daily_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate daily breaks
    """
    if info.width == 7:
        bymonthday = (1, 8, 15, 22)
    elif info.width == 14:
        bymonthday = (1, 15)
    else:
        bymonthday = range(1, 31, info.width)
    r = rrule(info.frequency, interval=1, dtstart=info.start, until=info.until, bymonthday=bymonthday)
    return r.between(info.start, info.until, True)