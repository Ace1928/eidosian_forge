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
def humanize_seconds(secs: int, prefix: str='', sep: str='', now: str='now', microseconds: bool=False) -> str:
    """Show seconds in human form.

    For example, 60 becomes "1 minute", and 7200 becomes "2 hours".

    Arguments:
        prefix (str): can be used to add a preposition to the output
            (e.g., 'in' will give 'in 1 second', but add nothing to 'now').
        now (str): Literal 'now'.
        microseconds (bool): Include microseconds.
    """
    secs = float(format(float(secs), '.2f'))
    for unit, divider, formatter in TIME_UNITS:
        if secs >= divider:
            w = secs / float(divider)
            return '{}{}{} {}'.format(prefix, sep, formatter(w), pluralize(w, unit))
    if microseconds and secs > 0.0:
        return '{prefix}{sep}{0:.2f} seconds'.format(secs, sep=sep, prefix=prefix)
    return now