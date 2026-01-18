import functools
import zoneinfo
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo
from asgiref.local import Local
from django.conf import settings
def get_current_timezone_name():
    """Return the name of the currently active time zone."""
    return _get_timezone_name(get_current_timezone())