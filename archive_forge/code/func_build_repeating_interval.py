import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def build_repeating_interval(cls, R=None, Rnn=None, interval=None):
    return RepeatingIntervalTuple(R, Rnn, interval)