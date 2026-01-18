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
@_invalidates_cache
def rdate(self, rdate):
    """ Include the given :py:class:`datetime` instance in the recurrence
            set generation. """
    self._rdate.append(rdate)