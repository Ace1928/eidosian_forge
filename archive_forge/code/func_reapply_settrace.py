from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback
def reapply_settrace():
    try:
        tracing_func = _last_tracing_func_thread_local.tracing_func
    except AttributeError:
        return
    else:
        SetTrace(tracing_func)