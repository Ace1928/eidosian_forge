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
def ll_fz_keep_colorspace(colorspace):
    """
    Low-level wrapper for `::fz_keep_colorspace()`.
    Increment the reference count for the colorspace.

    Returns the same pointer. Never throws an exception.
    """
    return _mupdf.ll_fz_keep_colorspace(colorspace)