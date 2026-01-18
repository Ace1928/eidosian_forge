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
def ll_fz_clone_path(path):
    """
    Low-level wrapper for `::fz_clone_path()`.
    Clone the data for a path.

    This is used in preference to fz_keep_path when a whole
    new copy of a path is required, rather than just a shared
    pointer. This probably indicates that the path is about to
    be modified.

    path: path to clone.

    Throws exceptions on failure to allocate.
    """
    return _mupdf.ll_fz_clone_path(path)