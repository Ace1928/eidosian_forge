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
def ll_fz_clone_pixmap(old):
    """
    Low-level wrapper for `::fz_clone_pixmap()`.
    Clone a pixmap, copying the pixels and associated data to new
    storage.

    The reference count of 'old' is unchanged.
    """
    return _mupdf.ll_fz_clone_pixmap(old)