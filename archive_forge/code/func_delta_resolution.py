from __future__ import annotations
import numbers
import os
import random
import sys
import time as _time
from calendar import monthrange
from datetime import date, datetime, timedelta
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from types import ModuleType
from typing import Any, Callable
from dateutil import tz as dateutil_tz
from dateutil.parser import isoparse
from kombu.utils.functional import reprcall
from kombu.utils.objects import cached_property
from .functional import dictfilter
from .text import pluralize
def delta_resolution(dt: datetime, delta: timedelta) -> datetime:
    """Round a :class:`~datetime.datetime` to the resolution of timedelta.

    If the :class:`~datetime.timedelta` is in days, the
    :class:`~datetime.datetime` will be rounded to the nearest days,
    if the :class:`~datetime.timedelta` is in hours the
    :class:`~datetime.datetime` will be rounded to the nearest hour,
    and so on until seconds, which will just return the original
    :class:`~datetime.datetime`.
    """
    delta = max(delta.total_seconds(), 0)
    resolutions = ((3, lambda x: x / 86400), (4, lambda x: x / 3600), (5, lambda x: x / 60))
    args = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    for res, predicate in resolutions:
        if predicate(delta) >= 1.0:
            return datetime(*args[:res], tzinfo=dt.tzinfo)
    return dt