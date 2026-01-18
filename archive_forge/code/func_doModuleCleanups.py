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
def doModuleCleanups():
    """Execute all module cleanup functions. Normally called for you after
    tearDownModule."""
    exceptions = []
    while _module_cleanups:
        function, args, kwargs = _module_cleanups.pop()
        try:
            function(*args, **kwargs)
        except Exception as exc:
            exceptions.append(exc)
    if exceptions:
        raise exceptions[0]