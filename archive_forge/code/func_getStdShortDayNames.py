import re, time, datetime
from .utils import isStr
def getStdShortDayNames():
    return [x[:3] for x in getStdDayNames()]