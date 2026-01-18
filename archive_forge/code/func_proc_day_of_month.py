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
def proc_day_of_month(d):
    try:
        expanded[2].index('*')
    except ValueError:
        days = DAYS[month - 1]
        if month == 2 and self.is_leap(year) is True:
            days += 1
        if 'l' in expanded[2] and days == d.day:
            return (False, d)
        if is_prev:
            days_in_prev_month = DAYS[(month - 2) % self.MONTHS_IN_YEAR]
            diff_day = nearest_diff_method(d.day, expanded[2], days_in_prev_month)
        else:
            diff_day = nearest_diff_method(d.day, expanded[2], days)
        if diff_day is not None and diff_day != 0:
            if is_prev:
                d += relativedelta(days=diff_day, hour=23, minute=59, second=59)
            else:
                d += relativedelta(days=diff_day, hour=0, minute=0, second=0)
            return (True, d)
    return (False, d)