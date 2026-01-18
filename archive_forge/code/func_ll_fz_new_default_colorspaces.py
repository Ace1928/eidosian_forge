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
def ll_fz_new_default_colorspaces():
    """
    Low-level wrapper for `::fz_new_default_colorspaces()`.
    Create a new default colorspace structure with values inherited
    from the context, and return a reference to it.

    These can be overridden using fz_set_default_xxxx.

    These should not be overridden while more than one caller has
    the reference for fear of race conditions.

    The caller should drop this reference once finished with it.
    """
    return _mupdf.ll_fz_new_default_colorspaces()