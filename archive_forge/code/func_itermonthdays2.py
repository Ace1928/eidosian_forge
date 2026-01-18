import sys
import datetime
import locale as _locale
from itertools import repeat
def itermonthdays2(self, year, month):
    """
        Like itermonthdates(), but will yield (day number, weekday number)
        tuples. For days outside the specified month the day number is 0.
        """
    for i, d in enumerate(self.itermonthdays(year, month), self.firstweekday):
        yield (d, i % 7)