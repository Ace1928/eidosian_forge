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
def hourly_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate hourly breaks
    """
    r = rrule(info.frequency, interval=1, dtstart=info.start, until=info.until, byhour=range(0, 24, info.width))
    return r.between(info.start, info.until, True)