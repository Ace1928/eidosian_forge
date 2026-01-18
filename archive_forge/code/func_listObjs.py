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
def listObjs(regex='Q', typ=None):
    """List all objects managed by python gc with class name matching regex.
    Finds 'Q...' classes by default."""
    if typ is not None:
        return [x for x in gc.get_objects() if isinstance(x, typ)]
    else:
        return [x for x in gc.get_objects() if re.match(regex, type(x).__name__)]