from collections import namedtuple
from math import floor, ceil
def inv_frac(value):
    """return inverse fractional part of x"""
    return 1 - (value - floor(value))