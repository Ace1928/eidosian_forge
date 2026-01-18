from __future__ import annotations
import re
from bisect import bisect, bisect_left
from collections import namedtuple
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo
from typing import Any, Callable, Mapping, Sequence
from kombu.utils.objects import cached_property
from celery import Celery
from . import current_app
from .utils.collections import AttributeDict
from .utils.time import (ffwd, humanize_seconds, localize, maybe_make_aware, maybe_timedelta, remaining, timezone,
def remaining_delta(self, last_run_at: datetime, tz: tzinfo | None=None, ffwd: type=ffwd) -> tuple[datetime, Any, datetime]:
    last_run_at = self.maybe_make_aware(last_run_at)
    now = self.maybe_make_aware(self.now())
    dow_num = last_run_at.isoweekday() % 7
    execute_this_date = last_run_at.month in self.month_of_year and last_run_at.day in self.day_of_month and (dow_num in self.day_of_week)
    execute_this_hour = execute_this_date and last_run_at.day == now.day and (last_run_at.month == now.month) and (last_run_at.year == now.year) and (last_run_at.hour in self.hour) and (last_run_at.minute < max(self.minute))
    if execute_this_hour:
        next_minute = min((minute for minute in self.minute if minute > last_run_at.minute))
        delta = ffwd(minute=next_minute, second=0, microsecond=0)
    else:
        next_minute = min(self.minute)
        execute_today = execute_this_date and last_run_at.hour < max(self.hour)
        if execute_today:
            next_hour = min((hour for hour in self.hour if hour > last_run_at.hour))
            delta = ffwd(hour=next_hour, minute=next_minute, second=0, microsecond=0)
        else:
            next_hour = min(self.hour)
            all_dom_moy = self._orig_day_of_month == '*' and self._orig_month_of_year == '*'
            if all_dom_moy:
                next_day = min([day for day in self.day_of_week if day > dow_num] or self.day_of_week)
                add_week = next_day == dow_num
                delta = ffwd(weeks=add_week and 1 or 0, weekday=(next_day - 1) % 7, hour=next_hour, minute=next_minute, second=0, microsecond=0)
            else:
                delta = self._delta_to_next(last_run_at, next_hour, next_minute)
    return (self.to_local(last_run_at), delta, self.to_local(now))