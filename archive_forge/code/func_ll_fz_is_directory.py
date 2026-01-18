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
def ll_fz_is_directory(path):
    """
    Low-level wrapper for `::fz_is_directory()`.
    Determine if a given path is a directory.

    In the case of the path not existing, or having no access
    we will return 0.
    """
    return _mupdf.ll_fz_is_directory(path)