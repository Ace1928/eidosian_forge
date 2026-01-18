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
def ll_fz_set_default_gray(default_cs, cs):
    """
    Low-level wrapper for `::fz_set_default_gray()`.
    Set new defaults within the default colorspace structure.

    New references are taken to the new default, and references to
    the old defaults dropped.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_set_default_gray(default_cs, cs)