import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def is_older_than(before, seconds):
    """Return True if before is older than seconds.

    .. versionchanged:: 1.7
       Accept datetime string with timezone information.
       Fix comparison with timezone aware datetime.
    """
    if isinstance(before, str):
        before = parse_isotime(before)
    before = normalize_time(before)
    return utcnow() - before > datetime.timedelta(seconds=seconds)