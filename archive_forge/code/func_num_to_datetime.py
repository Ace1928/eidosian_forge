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
def num_to_datetime(x: FloatArrayLike, tz: Optional[str | TzInfo]=None) -> NDArrayDatetime:
    """
    Convert any float array to numpy datetime64 array
    """
    tz = get_tzinfo(tz) or UTC
    return _from_ordinalf_np_vectorized(x, tz)