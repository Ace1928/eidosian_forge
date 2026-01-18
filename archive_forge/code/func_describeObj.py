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
def describeObj(obj, depth=4, path=None, ignore=None):
    """
    Trace all reference paths backward, printing a list of different ways this object can be accessed.
    Attempts to answer the question "who has a reference to this object"
    """
    if path is None:
        path = [obj]
    if ignore is None:
        ignore = {}
    ignore[id(sys._getframe())] = None
    ignore[id(path)] = None
    gc.collect()
    refs = gc.get_referrers(obj)
    ignore[id(refs)] = None
    printed = False
    for ref in refs:
        if id(ref) in ignore:
            continue
        if id(ref) in list(map(id, path)):
            print('Cyclic reference: ' + refPathString([ref] + path))
            printed = True
            continue
        newPath = [ref] + path
        if len(newPath) >= depth:
            refStr = refPathString(newPath)
            if '[_]' not in refStr:
                print(refStr)
            printed = True
        else:
            describeObj(ref, depth, newPath, ignore)
            printed = True
    if not printed:
        print('Dead end: ' + refPathString(path))