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
def listRedundantModules():
    """List modules that have been imported more than once via different paths."""
    mods = {}
    for name, mod in sys.modules.items():
        if not hasattr(mod, '__file__'):
            continue
        mfile = os.path.abspath(mod.__file__)
        if mfile[-1] == 'c':
            mfile = mfile[:-1]
        if mfile in mods:
            print('module at %s has 2 names: %s, %s' % (mfile, name, mods[mfile]))
        else:
            mods[mfile] = name