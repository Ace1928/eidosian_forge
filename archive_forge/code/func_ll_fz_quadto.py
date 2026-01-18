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
def ll_fz_quadto(path, x0, y0, x1, y1):
    """
    Low-level wrapper for `::fz_quadto()`.
    Append a 'quadto' command to an open path. (For a
    quadratic bezier).

    path: The path to modify.

    x0, y0: The control coordinates for the quadratic curve.

    x1, y1: The end coordinates for the quadratic curve.

    Throws exceptions on failure to allocate, or attempting to
    modify a packed path.
    """
    return _mupdf.ll_fz_quadto(path, x0, y0, x1, y1)