from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def floor_minute(d: datetime) -> datetime:
    """
    Round down to the start of the minute
    """
    if at_the_minute(d):
        return d
    return floor_hour(d).replace(minute=d.minute)