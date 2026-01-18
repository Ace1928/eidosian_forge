from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def ceil_second(d: datetime) -> datetime:
    """
    Round up to the start of the next minute
    """
    if at_the_second(d):
        return d
    return floor_second(d) + ONE_SECOND