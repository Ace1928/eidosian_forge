import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def instant(format=TIME_FORMAT, time_stamp=0):
    if time_stamp:
        return time.strftime(format, time.gmtime(time_stamp))
    else:
        return time.strftime(format, time.gmtime())