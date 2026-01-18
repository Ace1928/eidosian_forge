import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def marshall_now(now=None):
    """Make an rpc-safe datetime with microseconds.

    .. versionchanged:: 1.6
       Timezone information is now serialized instead of being stripped.
    """
    if not now:
        now = utcnow()
    d = dict(day=now.day, month=now.month, year=now.year, hour=now.hour, minute=now.minute, second=now.second, microsecond=now.microsecond)
    if now.tzinfo:
        tzname = now.tzinfo.tzname(None)
        d['tzname'] = 'UTC' if tzname == 'UTC+00:00' else tzname
    return d