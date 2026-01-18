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
def ll_fz_curvetov(path, x1, y1, x2, y2):
    """
    Low-level wrapper for `::fz_curvetov()`.
    Append a 'curvetov' command to an open path. (For a
    cubic bezier with the first control coordinate equal to
    the start point).

    path: The path to modify.

    x1, y1: The coordinates of the second control point for the
    curve.

    x2, y2: The end coordinates for the curve.

    Throws exceptions on failure to allocate, or attempting to
    modify a packed path.
    """
    return _mupdf.ll_fz_curvetov(path, x1, y1, x2, y2)