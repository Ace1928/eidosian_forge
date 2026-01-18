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
def ll_fz_lineto(path, x, y):
    """
    Low-level wrapper for `::fz_lineto()`.
    Append a 'lineto' command to an open path.

    path: The path to modify.

    x, y: The coordinate to line to.

    Throws exceptions on failure to allocate, or attempting to
    modify a packed path.
    """
    return _mupdf.ll_fz_lineto(path, x, y)