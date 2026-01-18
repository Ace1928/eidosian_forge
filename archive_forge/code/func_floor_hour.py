from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def floor_hour(d: datetime) -> datetime:
    """
    Round down to the start of the hour
    """
    if at_the_hour(d):
        return d
    return floor_day(d).replace(hour=d.hour)