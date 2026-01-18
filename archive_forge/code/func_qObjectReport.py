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
def qObjectReport(verbose=False):
    """Generate a report counting all QObjects and their types"""
    global qObjCache
    count = {}
    for obj in findObj('PyQt'):
        if isinstance(obj, QtCore.QObject):
            oid = id(obj)
            if oid not in QObjCache:
                QObjCache[oid] = typeStr(obj) + '  ' + obj.objectName()
                try:
                    QObjCache[oid] += '  ' + obj.parent().objectName()
                    QObjCache[oid] += '  ' + obj.text()
                except:
                    pass
            print('check obj', oid, str(QObjCache[oid]))
            if obj.parent() is None:
                walkQObjectTree(obj, count, verbose)
    typs = list(count.keys())
    typs.sort()
    for t in typs:
        print(count[t], '\t', t)