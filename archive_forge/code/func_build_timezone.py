import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def build_timezone(cls, negative=None, Z=None, hh=None, mm=None, name=''):
    return TimezoneTuple(negative, Z, hh, mm, name)