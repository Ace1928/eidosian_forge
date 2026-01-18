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
def proc_minute(d):
    try:
        expanded[0].index('*')
    except ValueError:
        diff_min = nearest_diff_method(d.minute, expanded[0], 60)
        if diff_min is not None and diff_min != 0:
            if is_prev:
                d += relativedelta(minutes=diff_min, second=59)
            else:
                d += relativedelta(minutes=diff_min, second=0)
            return (True, d)
    return (False, d)