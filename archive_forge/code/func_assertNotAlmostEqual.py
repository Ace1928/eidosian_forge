import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
def assertNotAlmostEqual(self, first, second, places=None, msg=None, delta=None):
    """Fail if the two objects are equal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           difference between the two objects is less than the given delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most significant digit).

           Objects that are equal automatically fail.
        """
    if delta is not None and places is not None:
        raise TypeError('specify delta or places not both')
    diff = abs(first - second)
    if delta is not None:
        if not first == second and diff > delta:
            return
        standardMsg = '%s == %s within %s delta (%s difference)' % (safe_repr(first), safe_repr(second), safe_repr(delta), safe_repr(diff))
    else:
        if places is None:
            places = 7
        if not first == second and round(diff, places) != 0:
            return
        standardMsg = '%s == %s within %r places' % (safe_repr(first), safe_repr(second), places)
    msg = self._formatMessage(msg, standardMsg)
    raise self.failureException(msg)