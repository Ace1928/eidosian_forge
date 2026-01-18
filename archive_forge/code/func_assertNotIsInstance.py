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
def assertNotIsInstance(self, obj, cls, msg=None):
    """Included for symmetry with assertIsInstance."""
    if isinstance(obj, cls):
        standardMsg = '%s is an instance of %r' % (safe_repr(obj), cls)
        self.fail(self._formatMessage(msg, standardMsg))