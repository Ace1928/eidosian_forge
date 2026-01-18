from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def floor_mid_year(d: datetime) -> datetime:
    """
    Round down half a year
    """
    _d_floor = floor_year(d)
    if d.month < 7:
        return _d_floor.replace(month=1)
    else:
        return _d_floor.replace(month=7)