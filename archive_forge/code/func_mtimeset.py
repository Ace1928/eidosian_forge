import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
from warnings import warn
from six import advance_iterator, integer_types
from six.moves import _thread, range
from ._common import weekday as weekdaybase
def mtimeset(self, hour, minute, second):
    tset = []
    rr = self.rrule
    for second in rr._bysecond:
        tset.append(datetime.time(hour, minute, second, tzinfo=rr._tzinfo))
    tset.sort()
    return tset