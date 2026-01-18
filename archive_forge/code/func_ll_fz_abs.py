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
def ll_fz_abs(f):
    """
    Low-level wrapper for `::fz_abs()`.
    Some standard math functions, done as static inlines for speed.
    People with compilers that do not adequately implement inline
    may like to reimplement these using macros.
    """
    return _mupdf.ll_fz_abs(f)