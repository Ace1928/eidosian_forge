import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def in_a_while(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0, format=TIME_FORMAT):
    """
    format of timedelta:
        timedelta([days[, seconds[, microseconds[, milliseconds[,
                    minutes[, hours[, weeks]]]]]]])
    """
    if format is None:
        format = TIME_FORMAT
    return time_in_a_while(days, seconds, microseconds, milliseconds, minutes, hours, weeks).strftime(format)