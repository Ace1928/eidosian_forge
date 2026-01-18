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
def assertIsNotNone(self, obj, msg=None):
    """Included for symmetry with assertIsNone."""
    if obj is None:
        standardMsg = 'unexpectedly None'
        self.fail(self._formatMessage(msg, standardMsg))