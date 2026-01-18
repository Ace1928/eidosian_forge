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
def ll_fz_realpath(path, resolved_path):
    """
    Low-level wrapper for `::fz_realpath()`.
    Resolve a path to an absolute file name.
    The resolved path buffer must be of at least PATH_MAX size.
    """
    return _mupdf.ll_fz_realpath(path, resolved_path)