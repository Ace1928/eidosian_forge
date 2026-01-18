import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
@classmethod
def no_cache(cls, key):
    obj = cls._new_instance(key)
    obj._from_cache = False
    return obj