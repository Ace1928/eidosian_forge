import logging
import sys
import types
import threading
import inspect
from functools import wraps
from itertools import chain
from numba.core import config
def chop(value):
    MAX_SIZE = 320
    s = repr(value)
    if len(s) > MAX_SIZE:
        return s[:MAX_SIZE] + '...' + s[-1]
    else:
        return s