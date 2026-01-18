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
def objectSize(obj, ignore=None, verbose=False, depth=0, recursive=False):
    """Guess how much memory an object is using"""
    ignoreTypes = ['MethodType', 'UnboundMethodType', 'BuiltinMethodType', 'FunctionType', 'BuiltinFunctionType']
    ignoreTypes = [getattr(types, key) for key in ignoreTypes if hasattr(types, key)]
    ignoreRegex = re.compile('(method-wrapper|Flag|ItemChange|Option|Mode)')
    if ignore is None:
        ignore = {}
    indent = '  ' * depth
    try:
        hash(obj)
        hsh = obj
    except:
        hsh = '%s:%d' % (str(type(obj)), id(obj))
    if hsh in ignore:
        return 0
    ignore[hsh] = 1
    try:
        size = sys.getsizeof(obj)
    except TypeError:
        size = 0
    if isinstance(obj, ndarray):
        try:
            size += len(obj.data)
        except:
            pass
    if recursive:
        if type(obj) in [list, tuple]:
            if verbose:
                print(indent + 'list:')
            for o in obj:
                s = objectSize(o, ignore=ignore, verbose=verbose, depth=depth + 1)
                if verbose:
                    print(indent + '  +', s)
                size += s
        elif isinstance(obj, dict):
            if verbose:
                print(indent + 'list:')
            for k in obj:
                s = objectSize(obj[k], ignore=ignore, verbose=verbose, depth=depth + 1)
                if verbose:
                    print(indent + '  +', k, s)
                size += s
        gc.collect()
        if verbose:
            print(indent + 'attrs:')
        for k in dir(obj):
            if k in ['__dict__']:
                continue
            o = getattr(obj, k)
            if type(o) in ignoreTypes:
                continue
            strtyp = str(type(o))
            if ignoreRegex.search(strtyp):
                continue
            refs = [r for r in gc.get_referrers(o) if type(r) != types.FrameType]
            if len(refs) == 1:
                s = objectSize(o, ignore=ignore, verbose=verbose, depth=depth + 1)
                size += s
                if verbose:
                    print(indent + '  +', k, s)
    return size