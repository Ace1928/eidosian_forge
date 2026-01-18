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
def printException(exctype, value, traceback):
    """Print an exception with its full traceback.
    
    Set `sys.excepthook = printException` to ensure that exceptions caught
    inside Qt signal handlers are printed with their full stack trace.
    """
    print(''.join(formatException(exctype, value, traceback, skip=1)))