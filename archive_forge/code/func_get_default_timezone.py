import functools
import zoneinfo
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo
from asgiref.local import Local
from django.conf import settings
@functools.lru_cache
def get_default_timezone():
    """
    Return the default time zone as a tzinfo instance.

    This is the time zone defined by settings.TIME_ZONE.
    """
    return zoneinfo.ZoneInfo(settings.TIME_ZONE)