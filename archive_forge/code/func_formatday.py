import sys
import datetime
import locale as _locale
from itertools import repeat
def formatday(self, day, weekday):
    """
        Return a day as a table cell.
        """
    if day == 0:
        return '<td class="%s">&nbsp;</td>' % self.cssclass_noday
    else:
        return '<td class="%s">%d</td>' % (self.cssclasses[weekday], day)