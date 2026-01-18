import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def has_stopped(self):
    """Returns True if the watch is in a stopped state."""
    return self._state == self._STOPPED