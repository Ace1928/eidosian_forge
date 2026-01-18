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
def ll_fz_unlock(lock):
    """
    Low-level wrapper for `::fz_unlock()`.
    Unlock one of the user supplied mutexes.
    """
    return _mupdf.ll_fz_unlock(lock)