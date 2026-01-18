import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def adjusted(self):
    """Return the adjusted exponent of self"""
    try:
        return self._exp + len(self._int) - 1
    except TypeError:
        return 0