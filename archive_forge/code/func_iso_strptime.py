import re
import datetime
def iso_strptime(time_str):
    x = RE_TIME.match(time_str)
    if not x:
        raise ValueError
    d = datetime.datetime(int(x.group('year')), int(x.group('month')), int(x.group('day')), int(x.group('hour')), int(x.group('minutes')), int(x.group('seconds')))
    if x.group('microseconds'):
        d = d.replace(microsecond=int(x.group('microseconds')))
    if x.group('tz_offset'):
        d = d.replace(tzinfo=TimeZone(x.group('tz_offset')))
    return d