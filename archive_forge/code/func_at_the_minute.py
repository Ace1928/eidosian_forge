from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def at_the_minute(d: datetime) -> bool:
    """
    Return True if the time of datetime is at the minute mark
    """
    t = d.time()
    return t.second == 0 and t.microsecond == 0