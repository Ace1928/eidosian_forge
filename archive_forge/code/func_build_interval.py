import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def build_interval(cls, start=None, end=None, duration=None):
    return IntervalTuple(start, end, duration)