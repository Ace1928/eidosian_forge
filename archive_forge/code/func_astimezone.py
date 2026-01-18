import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
def astimezone(self, *args, **kwargs):
    """tz -> convert to time in new timezone tz."""
    r = super(BaseTimestamp, self).astimezone(*args, **kwargs)
    return type(self)(r.year, r.month, r.day, r.hour, r.minute, r.second, r.microsecond, r.tzinfo)