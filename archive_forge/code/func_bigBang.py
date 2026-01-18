import re, time, datetime
from .utils import isStr
def bigBang():
    """return lower boundary as a NormalDate"""
    return NormalDate((-9999, 1, 1))