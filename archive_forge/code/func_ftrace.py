from __future__ import print_function
import contextlib
import cProfile
import gc
import inspect
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings
import weakref
from time import perf_counter
from numpy import ndarray
from .Qt import QT_LIB, QtCore
from .util import cprint
from .util.mutex import Mutex
def ftrace(func):
    """Decorator used for marking the beginning and end of function calls.
    Automatically indents nested calls.
    """

    def w(*args, **kargs):
        global __ftraceDepth
        pfx = '  ' * __ftraceDepth
        print(pfx + func.__name__ + ' start')
        __ftraceDepth += 1
        try:
            rv = func(*args, **kargs)
        finally:
            __ftraceDepth -= 1
        print(pfx + func.__name__ + ' done')
        return rv
    return w