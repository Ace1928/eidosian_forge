import sys
import datetime
import locale as _locale
from itertools import repeat
def itermonthdays4(self, year, month):
    """
        Like itermonthdates(), but will yield (year, month, day, day_of_week) tuples.
        Can be used for dates outside of datetime.date range.
        """
    for i, (y, m, d) in enumerate(self.itermonthdays3(year, month)):
        yield (y, m, d, (self.firstweekday + i) % 7)