import sys
import datetime
import locale as _locale
from itertools import repeat
def itermonthdays(self, year, month):
    """
        Like itermonthdates(), but will yield day numbers. For days outside
        the specified month the day number is 0.
        """
    day1, ndays = monthrange(year, month)
    days_before = (day1 - self.firstweekday) % 7
    yield from repeat(0, days_before)
    yield from range(1, ndays + 1)
    days_after = (self.firstweekday - day1 - ndays) % 7
    yield from repeat(0, days_after)