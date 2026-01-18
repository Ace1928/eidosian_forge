from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def floor_week(d: datetime) -> datetime:
    """
    Round down to the start of the week

    Week start on are on 1st, 8th, 15th and 22nd
    day of the month
    """
    if d.day < 8:
        day = 1
    elif d.day < 15:
        day = 8
    elif d.day < 22:
        day = 15
    else:
        day = 22
    return d.min.replace(d.year, d.month, day)