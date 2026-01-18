import re, time, datetime
from .utils import isStr
def lastDayOfMonth(self):
    """returns last day of the month as integer 28-31"""
    if self.isLeapYear():
        return _daysInMonthLeapYear[self.month() - 1]
    else:
        return _daysInMonthNormal[self.month() - 1]