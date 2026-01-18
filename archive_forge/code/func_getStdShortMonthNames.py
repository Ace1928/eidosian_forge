import re, time, datetime
from .utils import isStr
def getStdShortMonthNames():
    return [x[:3] for x in getStdMonthNames()]