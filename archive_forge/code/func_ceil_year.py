from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def ceil_year(d: datetime) -> datetime:
    """
    Round up to start of the year
    """
    _d_floor = floor_year(d)
    if d == _d_floor:
        return d
    else:
        return _d_floor + ONE_YEAR