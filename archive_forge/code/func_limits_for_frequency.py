from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def limits_for_frequency(self, freq: DateFrequency) -> TupleDatetime2:
    lookup = {DF.YEARLY: self.limits_year, DF.MONTHLY: self.limits_month, DF.DAILY: self.limits_day, DF.HOURLY: self.limits_hour, DF.MINUTELY: self.limits_minute, DF.SECONDLY: self.limits_second}
    try:
        return lookup[freq]()
    except KeyError:
        return self.limits