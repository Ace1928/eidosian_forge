from __future__ import annotations
from typing import TYPE_CHECKING
from isoduration.formatter.exceptions import DurationFormattingException
def check_global_sign(duration: Duration) -> int:
    is_date_zero = duration.date.years == 0 and duration.date.months == 0 and (duration.date.days == 0) and (duration.date.weeks == 0)
    is_time_zero = duration.time.hours == 0 and duration.time.minutes == 0 and (duration.time.seconds == 0)
    is_date_negative = duration.date.years <= 0 and duration.date.months <= 0 and (duration.date.days <= 0) and (duration.date.weeks <= 0)
    is_time_negative = duration.time.hours <= 0 and duration.time.minutes <= 0 and (duration.time.seconds <= 0)
    if not is_date_zero and (not is_time_zero):
        if is_date_negative and is_time_negative:
            return -1
    elif not is_date_zero:
        if is_date_negative:
            return -1
    elif not is_time_zero:
        if is_time_negative:
            return -1
    return +1