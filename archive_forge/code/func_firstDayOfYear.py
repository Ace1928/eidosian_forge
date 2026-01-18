import re, time, datetime
from .utils import isStr
def firstDayOfYear(year):
    """number of days to the first of the year, relative to Jan 1, 1900"""
    if not isinstance(year, int):
        msg = 'firstDayOfYear() expected integer, got %s' % type(year)
        raise NormalDateException(msg)
    if year == 0:
        raise NormalDateException('first day of year cannot be zero (0)')
    elif year < 0:
        firstDay = year * 365 + int((year - 1) / 4) - 693596
    else:
        leapAdjust = int((year + 3) / 4)
        if year > 1600:
            leapAdjust = leapAdjust - int((year + 99 - 1600) / 100) + int((year + 399 - 1600) / 400)
        firstDay = year * 365 + leapAdjust - 693963
        if year > 1582:
            firstDay = firstDay - 10
    return firstDay