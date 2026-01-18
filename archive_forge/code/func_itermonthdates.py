import sys
import datetime
import locale as _locale
from itertools import repeat
def itermonthdates(self, year, month):
    """
        Return an iterator for one month. The iterator will yield datetime.date
        values and will always iterate through complete weeks, so it will yield
        dates outside the specified month.
        """
    for y, m, d in self.itermonthdays3(year, month):
        yield datetime.date(y, m, d)