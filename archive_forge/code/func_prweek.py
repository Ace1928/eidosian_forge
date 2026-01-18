import sys
import datetime
import locale as _locale
from itertools import repeat
def prweek(self, theweek, width):
    """
        Print a single week (no newline).
        """
    print(self.formatweek(theweek, width), end='')