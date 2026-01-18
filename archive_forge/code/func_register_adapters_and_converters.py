import datetime
import time
import collections.abc
from _sqlite3 import *
def register_adapters_and_converters():

    def adapt_date(val):
        return val.isoformat()

    def adapt_datetime(val):
        return val.isoformat(' ')

    def convert_date(val):
        return datetime.date(*map(int, val.split(b'-')))

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
    register_adapter(datetime.date, adapt_date)
    register_adapter(datetime.datetime, adapt_datetime)
    register_converter('date', convert_date)
    register_converter('timestamp', convert_timestamp)