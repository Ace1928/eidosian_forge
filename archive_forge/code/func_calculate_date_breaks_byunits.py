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
def calculate_date_breaks_byunits(limits, units: DatetimeBreaksUnits, width: int, max_breaks: Optional[int]=None, tz: Optional[TzInfo]=None) -> Sequence[datetime]:
    """
    Calcuate date breaks using appropriate units
    """
    timely_name = f'{units.upper()}LY'
    if timely_name in ('DAYLY', 'WEEKLY'):
        timely_name = 'DAILY'
    freq = getattr(DF, timely_name)
    start, until = expand_datetime_limits(limits, width, units)
    if units == 'week':
        width *= 7
    info = date_breaks_info(freq, n=-1, width=width, start=start, until=until, tz=tz)
    lookup = {'year': rrulely_breaks, 'month': rrulely_breaks, 'week': rrulely_breaks, 'day': rrulely_breaks, 'hour': rrulely_breaks, 'minute': rrulely_breaks, 'second': rrulely_breaks, 'microsecond': microsecondly_breaks}
    return lookup[units](info)