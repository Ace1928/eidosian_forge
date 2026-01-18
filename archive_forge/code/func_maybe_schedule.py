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
def maybe_schedule(s: int | float | timedelta | BaseSchedule, relative: bool=False, app: Celery | None=None) -> float | timedelta | BaseSchedule:
    """Return schedule from number, timedelta, or actual schedule."""
    if s is not None:
        if isinstance(s, (float, int)):
            s = timedelta(seconds=s)
        if isinstance(s, timedelta):
            return schedule(s, relative, app=app)
        else:
            s.app = app
    return s