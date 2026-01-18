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
def ll_fz_moveto(path, x, y):
    """
    Low-level wrapper for `::fz_moveto()`.
    Append a 'moveto' command to a path.
    This 'opens' a path.

    path: The path to modify.

    x, y: The coordinate to move to.

    Throws exceptions on failure to allocate, or attempting to
    modify a packed path.
    """
    return _mupdf.ll_fz_moveto(path, x, y)