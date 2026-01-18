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
def ll_fz_close_device(dev):
    """
    Low-level wrapper for `::fz_close_device()`.
    Signal the end of input, and flush any buffered output.
    This is NOT called implicitly on fz_drop_device. This
    may throw exceptions.
    """
    return _mupdf.ll_fz_close_device(dev)