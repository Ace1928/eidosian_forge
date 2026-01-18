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
def objString(obj):
    """Return a short but descriptive string for any object"""
    try:
        if type(obj) in [int, float]:
            return str(obj)
        elif isinstance(obj, dict):
            if len(obj) > 5:
                return '<dict {%s,...}>' % ','.join(list(obj.keys())[:5])
            else:
                return '<dict {%s}>' % ','.join(list(obj.keys()))
        elif isinstance(obj, str):
            if len(obj) > 50:
                return '"%s..."' % obj[:50]
            else:
                return obj[:]
        elif isinstance(obj, ndarray):
            return '<ndarray %s %s>' % (str(obj.dtype), str(obj.shape))
        elif hasattr(obj, '__len__'):
            if len(obj) > 5:
                return '<%s [%s,...]>' % (type(obj).__name__, ','.join([type(o).__name__ for o in obj[:5]]))
            else:
                return '<%s [%s]>' % (type(obj).__name__, ','.join([type(o).__name__ for o in obj]))
        else:
            return '<%s %s>' % (type(obj).__name__, obj.__class__.__name__)
    except:
        return str(type(obj))