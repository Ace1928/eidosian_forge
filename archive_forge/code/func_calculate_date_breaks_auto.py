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
def calculate_date_breaks_auto(limits, n: int=5) -> Sequence[datetime]:
    """
    Calcuate date breaks using appropriate units
    """
    info = calculate_date_breaks_info(limits, n=n)
    lookup = {DF.YEARLY: yearly_breaks, DF.MONTHLY: monthly_breaks, DF.DAILY: daily_breaks, DF.HOURLY: hourly_breaks, DF.MINUTELY: minutely_breaks, DF.SECONDLY: secondly_breaks, DF.MICROSECONDLY: microsecondly_breaks}
    return lookup[info.frequency](info)