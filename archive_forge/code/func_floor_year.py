from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def floor_year(d: datetime) -> datetime:
    """
    Round down to the start of the year
    """
    return d.min.replace(d.year)