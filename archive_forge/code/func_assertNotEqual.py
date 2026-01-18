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
def assertNotEqual(self, first, second, msg=None):
    """Fail if the two objects are equal as determined by the '!='
           operator.
        """
    if not first != second:
        msg = self._formatMessage(msg, '%s == %s' % (safe_repr(first), safe_repr(second)))
        raise self.failureException(msg)