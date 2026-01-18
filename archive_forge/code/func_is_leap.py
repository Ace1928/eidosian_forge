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
def is_leap(self, year):
    if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
        return True
    else:
        return False