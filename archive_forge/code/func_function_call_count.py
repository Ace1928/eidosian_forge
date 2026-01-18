from __future__ import annotations
import collections
import contextlib
import os
import platform
import pstats
import re
import sys
from . import config
from .util import gc_collect
from ..util import has_compiled_ext
def function_call_count(variance=0.05, times=1, warmup=0):
    """Assert a target for a test case's function call count.

    The main purpose of this assertion is to detect changes in
    callcounts for various functions - the actual number is not as important.
    Callcounts are stored in a file keyed to Python version and OS platform
    information.  This file is generated automatically for new tests,
    and versioned so that unexpected changes in callcounts will be detected.

    """
    from sqlalchemy.util import decorator

    @decorator
    def wrap(fn, *args, **kw):
        for warm in range(warmup):
            fn(*args, **kw)
        timerange = range(times)
        with count_functions(variance=variance):
            for time in timerange:
                rv = fn(*args, **kw)
            return rv
    return wrap