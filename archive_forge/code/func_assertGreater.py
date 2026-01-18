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
def assertGreater(self, a, b, msg=None):
    """Just like self.assertTrue(a > b), but with a nicer default message."""
    if not a > b:
        standardMsg = '%s not greater than %s' % (safe_repr(a), safe_repr(b))
        self.fail(self._formatMessage(msg, standardMsg))