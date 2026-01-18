import datetime
import time
import collections.abc
from _sqlite3 import *
def convert_timestamp(val):
    datepart, timepart = val.split(b' ')
    year, month, day = map(int, datepart.split(b'-'))
    timepart_full = timepart.split(b'.')
    hours, minutes, seconds = map(int, timepart_full[0].split(b':'))
    if len(timepart_full) == 2:
        microseconds = int('{:0<6.6}'.format(timepart_full[1].decode()))
    else:
        microseconds = 0
    val = datetime.datetime(year, month, day, hours, minutes, seconds, microseconds)
    return val