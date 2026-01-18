from __future__ import annotations
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from functools import lru_cache
import re
from typing import Any
from ._types import ParseFloat
def match_to_localtime(match: re.Match) -> time:
    hour_str, minute_str, sec_str, micros_str = match.groups()
    micros = int(micros_str.ljust(6, '0')) if micros_str else 0
    return time(int(hour_str), int(minute_str), int(sec_str), micros)