import sys
import datetime
import locale as _locale
from itertools import repeat
def monthdayscalendar(self, year, month):
    """
        Return a matrix representing a month's calendar.
        Each row represents a week; days outside this month are zero.
        """
    days = list(self.itermonthdays(year, month))
    return [days[i:i + 7] for i in range(0, len(days), 7)]