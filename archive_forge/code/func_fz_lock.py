from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_lock(lock):
    """
    Class-aware wrapper for `::fz_lock()`.
    	Lock one of the user supplied mutexes.
    """
    return _mupdf.fz_lock(lock)