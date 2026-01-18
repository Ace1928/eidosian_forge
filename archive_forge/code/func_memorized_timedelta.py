from datetime import datetime, timedelta, tzinfo
from bisect import bisect_right
import pytz
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
def memorized_timedelta(seconds):
    """Create only one instance of each distinct timedelta"""
    try:
        return _timedelta_cache[seconds]
    except KeyError:
        delta = timedelta(seconds=seconds)
        _timedelta_cache[seconds] = delta
        return delta