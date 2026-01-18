import functools
import zoneinfo
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo
from asgiref.local import Local
from django.conf import settings
def make_aware(value, timezone=None):
    """Make a naive datetime.datetime in a given time zone aware."""
    if timezone is None:
        timezone = get_current_timezone()
    if is_aware(value):
        raise ValueError('make_aware expects a naive datetime, got %s' % value)
    return value.replace(tzinfo=timezone)