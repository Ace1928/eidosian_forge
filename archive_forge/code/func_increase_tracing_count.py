from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback
def increase_tracing_count():
    set_trace = TracingFunctionHolder._original_tracing or sys.settrace
    set_trace(dummy_trace)
    proceed.release()