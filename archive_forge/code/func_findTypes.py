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
def findTypes(self, refs, regex):
    allObjs = get_all_objects()
    objs = []
    r = re.compile(regex)
    for k in refs:
        if r.search(self.objTypes[k]):
            objs.append(self.lookup(k, refs[k], allObjs))
    return objs