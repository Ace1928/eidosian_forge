import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def range_check_time(cls, hh=None, mm=None, ss=None, tz=None, rangedict=None):
    midnight = False
    if rangedict is None:
        rangedict = cls.TIME_RANGE_DICT
    if 'hh' in rangedict:
        try:
            hh = rangedict['hh'].rangefunc(hh, rangedict['hh'])
        except HoursOutOfBoundsError as e:
            if float(hh) > 24 and float(hh) < 25:
                raise MidnightBoundsError('Hour 24 may only represent midnight.')
            raise e
    if 'mm' in rangedict:
        mm = rangedict['mm'].rangefunc(mm, rangedict['mm'])
    if 'ss' in rangedict:
        ss = rangedict['ss'].rangefunc(ss, rangedict['ss'])
    if hh is not None and hh == 24:
        midnight = True
    if midnight is True and (mm is not None and mm != 0 or (ss is not None and ss != 0)):
        raise MidnightBoundsError('Hour 24 may only represent midnight.')
    if cls.LEAP_SECONDS_SUPPORTED is True:
        if hh != 23 and mm != 59 and (ss == 60):
            raise cls.TIME_SS_LIMIT.rangeexception(cls.TIME_SS_LIMIT.rangeerrorstring)
    else:
        if hh == 23 and mm == 59 and (ss == 60):
            raise LeapSecondError('Leap seconds are not supported.')
        if ss == 60:
            raise cls.TIME_SS_LIMIT.rangeexception(cls.TIME_SS_LIMIT.rangeerrorstring)
    return (hh, mm, ss, tz)