from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def at_the_second(d: datetime) -> bool:
    """
    Return True if the time of datetime is at the second mark
    """
    return d.time().microsecond == 0