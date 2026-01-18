from __future__ import absolute_import, print_function, division
import traceback as _traceback
import copy
import math
import re
import sys
import inspect
from time import time
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
import calendar
import binascii
import random
import pytz  # noqa
def croniter_range(start, stop, expr_format, ret_type=None, day_or=True, exclude_ends=False, _croniter=None):
    """
    Generator that provides all times from start to stop matching the given cron expression.
    If the cron expression matches either 'start' and/or 'stop', those times will be returned as
    well unless 'exclude_ends=True' is passed.

    You can think of this function as sibling to the builtin range function for datetime objects.
    Like range(start,stop,step), except that here 'step' is a cron expression.
    """
    _croniter = _croniter or croniter
    auto_rt = datetime.datetime
    if type(start) is not type(stop) and (not (isinstance(start, type(stop)) or isinstance(stop, type(start)))):
        raise CroniterBadTypeRangeError('The start and stop must be same type.  {0} != {1}'.format(type(start), type(stop)))
    if isinstance(start, (float, int)):
        start, stop = (datetime.datetime.fromtimestamp(t, tzutc()).replace(tzinfo=None) for t in (start, stop))
        auto_rt = float
    if ret_type is None:
        ret_type = auto_rt
    if not exclude_ends:
        ms1 = relativedelta(microseconds=1)
        if start < stop:
            start -= ms1
            stop += ms1
        else:
            start += ms1
            stop -= ms1
    year_span = math.floor(abs(stop.year - start.year)) + 1
    ic = _croniter(expr_format, start, ret_type=datetime.datetime, day_or=day_or, max_years_between_matches=year_span)
    if start < stop:

        def cont(v):
            return v < stop
        step = ic.get_next
    else:

        def cont(v):
            return v > stop
        step = ic.get_prev
    try:
        dt = step()
        while cont(dt):
            if ret_type is float:
                yield ic.get_current(float)
            else:
                yield dt
            dt = step()
    except CroniterBadDateError:
        return