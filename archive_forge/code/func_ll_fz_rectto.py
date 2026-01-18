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
def ll_fz_rectto(path, x0, y0, x1, y1):
    """
    Low-level wrapper for `::fz_rectto()`.
    Append a 'rectto' command to an open path.

    The rectangle is equivalent to:
    	moveto x0 y0
    	lineto x1 y0
    	lineto x1 y1
    	lineto x0 y1
    	closepath

    path: The path to modify.

    x0, y0: First corner of the rectangle.

    x1, y1: Second corner of the rectangle.

    Throws exceptions on failure to allocate, or attempting to
    modify a packed path.
    """
    return _mupdf.ll_fz_rectto(path, x0, y0, x1, y1)