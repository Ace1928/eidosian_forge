import re, time, datetime
from .utils import isStr
def monthAbbrev(self):
    """returns month as a 3-character abbreviation, i.e. Jan, Feb, etc."""
    return self.__month_name__[self.month() - 1][:3]