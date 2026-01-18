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
def refPathString(chain):
    """Given a list of adjacent objects in a reference path, print the 'natural' path
    names (ie, attribute names, keys, and indexes) that follow from one object to the next ."""
    s = objString(chain[0])
    i = 0
    while i < len(chain) - 1:
        i += 1
        o1 = chain[i - 1]
        o2 = chain[i]
        cont = False
        if isinstance(o1, list) or isinstance(o1, tuple):
            if any((o2 is x for x in o1)):
                s += '[%d]' % o1.index(o2)
                continue
        if isinstance(o2, dict) and hasattr(o1, '__dict__') and (o2 == o1.__dict__):
            i += 1
            if i >= len(chain):
                s += '.__dict__'
                continue
            o3 = chain[i]
            for k in o2:
                if o2[k] is o3:
                    s += '.%s' % k
                    cont = True
                    continue
        if isinstance(o1, dict):
            try:
                if o2 in o1:
                    s += '[key:%s]' % objString(o2)
                    continue
            except TypeError:
                pass
            for k in o1:
                if o1[k] is o2:
                    s += '[%s]' % objString(k)
                    cont = True
                    continue
        if cont:
            continue
        s += ' ? '
        sys.stdout.flush()
    return s