import sys
import datetime
import locale as _locale
from itertools import repeat
def itermonthdays3(self, year, month):
    """
        Like itermonthdates(), but will yield (year, month, day) tuples.  Can be
        used for dates outside of datetime.date range.
        """
    day1, ndays = monthrange(year, month)
    days_before = (day1 - self.firstweekday) % 7
    days_after = (self.firstweekday - day1 - ndays) % 7
    y, m = _prevmonth(year, month)
    end = _monthlen(y, m) + 1
    for d in range(end - days_before, end):
        yield (y, m, d)
    for d in range(1, ndays + 1):
        yield (year, month, d)
    y, m = _nextmonth(year, month)
    for d in range(1, days_after + 1):
        yield (y, m, d)