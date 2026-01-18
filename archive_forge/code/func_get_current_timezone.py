import functools
import zoneinfo
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo
from asgiref.local import Local
from django.conf import settings
def get_current_timezone():
    """Return the currently active time zone as a tzinfo instance."""
    return getattr(_active, 'value', get_default_timezone())