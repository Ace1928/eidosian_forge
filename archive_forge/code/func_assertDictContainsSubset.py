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
def assertDictContainsSubset(self, subset, dictionary, msg=None):
    """Checks whether dictionary is a superset of subset."""
    warnings.warn('assertDictContainsSubset is deprecated', DeprecationWarning)
    missing = []
    mismatched = []
    for key, value in subset.items():
        if key not in dictionary:
            missing.append(key)
        elif value != dictionary[key]:
            mismatched.append('%s, expected: %s, actual: %s' % (safe_repr(key), safe_repr(value), safe_repr(dictionary[key])))
    if not (missing or mismatched):
        return
    standardMsg = ''
    if missing:
        standardMsg = 'Missing: %s' % ','.join((safe_repr(m) for m in missing))
    if mismatched:
        if standardMsg:
            standardMsg += '; '
        standardMsg += 'Mismatched values: %s' % ','.join(mismatched)
    self.fail(self._formatMessage(msg, standardMsg))