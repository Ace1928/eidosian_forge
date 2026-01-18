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
def enableFaulthandler():
    """ Enable faulthandler for all threads. 
    
    If the faulthandler package is available, this function disables and then 
    re-enables fault handling for all threads (this is necessary to ensure any
    new threads are handled correctly), and returns True.

    If faulthandler is not available, then returns False.
    """
    try:
        import faulthandler
        faulthandler.disable()
        faulthandler.enable(all_threads=True)
        return True
    except ImportError:
        return False