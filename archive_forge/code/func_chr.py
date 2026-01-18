from __future__ import unicode_literals
import inspect
import sys
import math
import numbers
from future.utils import PY2, PY3, exec_
def chr(i):
    """
        Return a byte-string of one character with ordinal i; 0 <= i <= 256
        """
    return oldstr(bytes((i,)))