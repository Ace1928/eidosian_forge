import sys
import datetime
import locale as _locale
from itertools import repeat
def iterweekdays(self):
    """
        Return an iterator for one week of weekday numbers starting with the
        configured first one.
        """
    for i in range(self.firstweekday, self.firstweekday + 7):
        yield (i % 7)