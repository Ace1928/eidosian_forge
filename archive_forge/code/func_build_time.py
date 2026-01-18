import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def build_time(cls, hh=None, mm=None, ss=None, tz=None):
    return TimeTuple(hh, mm, ss, tz)