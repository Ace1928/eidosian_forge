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
def assertNotIn(self, member, container, msg=None):
    """Just like self.assertTrue(a not in b), but with a nicer default message."""
    if member in container:
        standardMsg = '%s unexpectedly found in %s' % (safe_repr(member), safe_repr(container))
        self.fail(self._formatMessage(msg, standardMsg))