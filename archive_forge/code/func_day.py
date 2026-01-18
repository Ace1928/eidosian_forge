import re, time, datetime
from .utils import isStr
def day(self):
    """return the day as integer 1-31"""
    return int(repr(self.normalDate)[-2:])