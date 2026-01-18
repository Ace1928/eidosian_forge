import re, time, datetime
from .utils import isStr
def dayOfYear(self):
    """day of year"""
    if self.isLeapYear():
        daysByMonth = _daysInMonthLeapYear
    else:
        daysByMonth = _daysInMonthNormal
    priorMonthDays = 0
    for m in range(self.month() - 1):
        priorMonthDays = priorMonthDays + daysByMonth[m]
    return self.day() + priorMonthDays