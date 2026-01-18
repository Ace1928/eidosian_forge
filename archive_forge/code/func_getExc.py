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
def getExc(indent=4, prefix='|  ', skip=1):
    lines = formatException(*sys.exc_info(), skip=skip)
    lines2 = []
    for l in lines:
        lines2.extend(l.strip('\n').split('\n'))
    lines3 = [' ' * indent + prefix + l for l in lines2]
    return '\n'.join(lines3)