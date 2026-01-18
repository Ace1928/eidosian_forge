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
def listQThreads():
    """Prints Thread IDs (Qt's, not OS's) for all QThreads."""
    thr = findObj('[Tt]hread')
    thr = [t for t in thr if isinstance(t, QtCore.QThread)]
    try:
        from PyQt5 import sip
    except ImportError:
        import sip
    for t in thr:
        print('--> ', t)
        print('     Qt ID: 0x%x' % sip.unwrapinstance(t))