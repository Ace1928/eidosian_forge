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
def all_next(self, ret_type=None):
    """Generator of all consecutive dates. Can be used instead of
        implicit call to __iter__, whenever non-default
        'ret_type' has to be specified.
        """
    try:
        while True:
            self._is_prev = False
            yield self._get_next(ret_type or self._ret_type)
    except CroniterBadDateError:
        if self._max_years_btw_matches_explicitly_set:
            return
        else:
            raise