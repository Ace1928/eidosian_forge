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
def ll_fz_drop_colorspace(colorspace):
    """
    Low-level wrapper for `::fz_drop_colorspace()`.
    Drops a reference to the colorspace.

    When the reference count reaches zero, the colorspace is
    destroyed.
    """
    return _mupdf.ll_fz_drop_colorspace(colorspace)