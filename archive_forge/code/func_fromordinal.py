import time as _time
import math as _math
import sys
from operator import index as _index
@classmethod
def fromordinal(cls, n):
    """Construct a date from a proleptic Gregorian ordinal.

        January 1 of year 1 is day 1.  Only the year, month and day are
        non-zero in the result.
        """
    y, m, d = _ord2ymd(n)
    return cls(y, m, d)