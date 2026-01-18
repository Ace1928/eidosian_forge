import re, time, datetime
from .utils import isStr
def formatUS(self):
    """return date as string in common US format: MM/DD/YY"""
    d = self.__repr__()
    return '%s/%s/%s' % (d[-4:-2], d[-2:], d[-6:-4])