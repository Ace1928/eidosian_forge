from __future__ import annotations
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from functools import lru_cache
import re
from typing import Any
from ._types import ParseFloat
def match_to_datetime(match: re.Match) -> datetime | date:
    """Convert a `RE_DATETIME` match to `datetime.datetime` or `datetime.date`.

    Raises ValueError if the match does not correspond to a valid date
    or datetime.
    """
    year_str, month_str, day_str, hour_str, minute_str, sec_str, micros_str, zulu_time, offset_sign_str, offset_hour_str, offset_minute_str = match.groups()
    year, month, day = (int(year_str), int(month_str), int(day_str))
    if hour_str is None:
        return date(year, month, day)
    hour, minute, sec = (int(hour_str), int(minute_str), int(sec_str))
    micros = int(micros_str.ljust(6, '0')) if micros_str else 0
    if offset_sign_str:
        tz: tzinfo | None = cached_tz(offset_hour_str, offset_minute_str, offset_sign_str)
    elif zulu_time:
        tz = timezone.utc
    else:
        tz = None
    return datetime(year, month, day, hour, minute, sec, micros, tzinfo=tz)