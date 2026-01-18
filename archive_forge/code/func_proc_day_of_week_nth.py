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
def proc_day_of_week_nth(d):
    if '*' in nth_weekday_of_month:
        s = nth_weekday_of_month['*']
        for i in range(0, 7):
            if i in nth_weekday_of_month:
                nth_weekday_of_month[i].update(s)
            else:
                nth_weekday_of_month[i] = s
        del nth_weekday_of_month['*']
    candidates = []
    for wday, nth in nth_weekday_of_month.items():
        c = self._get_nth_weekday_of_month(d.year, d.month, wday)
        for n in nth:
            if n == 'l':
                candidate = c[-1]
            elif len(c) < n:
                continue
            else:
                candidate = c[n - 1]
            if is_prev and candidate <= d.day or (not is_prev and d.day <= candidate):
                candidates.append(candidate)
    if not candidates:
        if is_prev:
            d += relativedelta(days=-d.day, hour=23, minute=59, second=59)
        else:
            days = DAYS[month - 1]
            if month == 2 and self.is_leap(year) is True:
                days += 1
            d += relativedelta(days=days - d.day + 1, hour=0, minute=0, second=0)
        return (True, d)
    candidates.sort()
    diff_day = (candidates[-1] if is_prev else candidates[0]) - d.day
    if diff_day != 0:
        if is_prev:
            d += relativedelta(days=diff_day, hour=23, minute=59, second=59)
        else:
            d += relativedelta(days=diff_day, hour=0, minute=0, second=0)
        return (True, d)
    return (False, d)