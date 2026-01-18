from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def ceil_day(d: datetime) -> datetime:
    """
    Round up to the start of the next day
    """
    return floor_day(d) + ONE_DAY if has_time(d) else d