import re, time, datetime
from .utils import isStr
def setDay(self, day):
    """set the day of the month"""
    maxDay = self.lastDayOfMonth()
    if day < 1 or day > maxDay:
        msg = 'day is outside of range 1 to %d' % maxDay
        raise NormalDateException(msg)
    y, m, d = self.toTuple()
    self.setNormalDate((y, m, day))