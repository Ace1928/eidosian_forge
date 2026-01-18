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
def proc_hour(d):
    try:
        expanded[1].index('*')
    except ValueError:
        diff_hour = nearest_diff_method(d.hour, expanded[1], 24)
        if diff_hour is not None and diff_hour != 0:
            if is_prev:
                d += relativedelta(hours=diff_hour, minute=59, second=59)
            else:
                d += relativedelta(hours=diff_hour, minute=0, second=0)
            return (True, d)
    return (False, d)