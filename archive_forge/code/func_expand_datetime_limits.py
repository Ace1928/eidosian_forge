from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def expand_datetime_limits(limits: TupleDatetime2, width: int, units: DatetimeBreaksUnits) -> TupleDatetime2:
    ival = Interval(*limits)
    if units == 'year':
        start, end = ival.limits_year()
        span = ival.y_wide
    elif units == 'month':
        start, end = ival.limits_month()
        span = ival.M_wide
    elif units == 'week':
        start, end = ival.limits_week()
        span = ival.w_wide
    elif units == 'day':
        start, end = ival.limits_day()
        span = ival.d_wide
    elif units == 'hour':
        start, end = ival.limits_hour()
        span = ival.h_wide
    elif units == 'minute':
        start, end = ival.limits_minute()
        span = ival.m_wide
    elif units == 'second':
        start, end = ival.limits_second()
        span = ival.s
    else:
        return limits
    new_span = math.ceil(span / width) * width
    if units == 'week':
        units, new_span = ('day', new_span * 7)
    end = start + relativedelta(None, None, **{f'{units}s': new_span})
    if units == 'year':
        limits_orig = (limits[0].year, limits[1].year)
        y1, y2 = shift_limits_down((start.year, end.year), limits_orig, width)
        start = start.replace(y1)
        end = end.replace(y2)
    return (start, end)