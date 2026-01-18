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
def findRefPath(startObj, endObj, maxLen=8, restart=True, seen=None, path=None, ignore=None):
    """Determine all paths of object references from startObj to endObj"""
    refs = []
    if path is None:
        path = [endObj]
    if ignore is None:
        ignore = {}
    if seen is None:
        seen = {}
    ignore[id(sys._getframe())] = None
    ignore[id(path)] = None
    ignore[id(seen)] = None
    prefix = ' ' * (8 - maxLen)
    prefix += ' '
    if restart:
        seen.clear()
    gc.collect()
    newRefs = [r for r in gc.get_referrers(endObj) if id(r) not in ignore]
    ignore[id(newRefs)] = None
    for r in newRefs:
        if type(r).__name__ in ['frame', 'function', 'listiterator']:
            continue
        try:
            if any((r is x for x in path)):
                continue
        except:
            print(r)
            print(path)
            raise
        if r is startObj:
            refs.append([r])
            print(refPathString([startObj] + path))
            continue
        if maxLen == 0:
            continue
        tree = None
        try:
            cache = seen[id(r)]
            if cache[0] >= maxLen:
                tree = cache[1]
                for p in tree:
                    print(refPathString(p + path))
        except KeyError:
            pass
        ignore[id(tree)] = None
        if tree is None:
            tree = findRefPath(startObj, r, maxLen - 1, restart=False, path=[r] + path, ignore=ignore)
            seen[id(r)] = [maxLen, tree]
        if len(tree) == 0:
            continue
        else:
            for p in tree:
                refs.append(p + [r])
    return refs