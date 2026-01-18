from abc import abstractmethod
import math
import operator
import re
import datetime
from calendar import isleap
from decimal import Decimal, Context
from typing import cast, Any, Callable, Dict, Optional, Tuple, Union
from ..helpers import MONTH_DAYS_LEAP, MONTH_DAYS, DAYS_IN_4Y, \
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
@classmethod
def fromdelta(cls, delta: datetime.timedelta, adjust_timezone: bool=False) -> 'OrderedDateTime':
    """
        Creates an XSD dateTime/date instance from a datetime.timedelta related to
        0001-01-01T00:00:00 CE. In case of a date the time part is not counted.

        :param delta: a datetime.timedelta instance.
        :param adjust_timezone: if `True` adjusts the timezone of Date objects         with eventually present hours and minutes.
        """
    try:
        dt = datetime.datetime(1, 1, 1) + delta
    except OverflowError:
        days = delta.days
        if days > 0:
            y400, days = divmod(days, DAYS_IN_400Y)
            y100, days = divmod(days, DAYS_IN_100Y)
            y4, days = divmod(days, DAYS_IN_4Y)
            y1, days = divmod(days, 365)
            year = y400 * 400 + y100 * 100 + y4 * 4 + y1 + 1
            if y1 == 4 or y100 == 4:
                year -= 1
                days = 365
            td = datetime.timedelta(days=days, seconds=delta.seconds, microseconds=delta.microseconds)
            dt = datetime.datetime(4 if isleap(year) else 6, 1, 1) + td
        elif days >= -366:
            year = -1
            td = datetime.timedelta(days=days, seconds=delta.seconds, microseconds=delta.microseconds)
            dt = datetime.datetime(5, 1, 1) + td
        else:
            days = -days - 366
            y400, days = divmod(days, DAYS_IN_400Y)
            y100, days = divmod(days, DAYS_IN_100Y)
            y4, days = divmod(days, DAYS_IN_4Y)
            y1, days = divmod(days, 365)
            year = -y400 * 400 - y100 * 100 - y4 * 4 - y1 - 2
            if y1 == 4 or y100 == 4:
                year += 1
                days = 365
            td = datetime.timedelta(days=-days, seconds=delta.seconds, microseconds=delta.microseconds)
            if not td:
                dt = datetime.datetime(4 if isleap(year + 1) else 6, 1, 1)
                year += 1
            else:
                dt = datetime.datetime(5 if isleap(year + 1) else 7, 1, 1) + td
    else:
        year = dt.year
    if issubclass(cls, Date10):
        if adjust_timezone and (dt.hour or dt.minute):
            assert dt.tzinfo is None
            hour, minute = (dt.hour, dt.minute)
            if hour < 14 or (hour == 14 and minute == 0):
                tz = Timezone(datetime.timedelta(hours=-hour, minutes=-minute))
                dt = dt.replace(tzinfo=tz)
            else:
                tz = Timezone(datetime.timedelta(hours=-dt.hour + 24, minutes=-minute))
                dt = dt.replace(tzinfo=tz)
                dt += datetime.timedelta(days=1)
        return cls(year, dt.month, dt.day, tzinfo=dt.tzinfo)
    return cls(year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, dt.tzinfo)