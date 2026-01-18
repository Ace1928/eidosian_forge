import re, time, datetime
from .utils import isStr
def setYear(self, year):
    if year == 0:
        raise NormalDateException('cannot set year to zero')
    elif year < -9999:
        raise NormalDateException('year cannot be less than -9999')
    elif year > 9999:
        raise NormalDateException('year cannot be greater than 9999')
    y, m, d = self.toTuple()
    self.setNormalDate((year, m, d))