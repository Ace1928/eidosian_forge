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
def ll_fz_write_string(out, s):
    """
    Low-level wrapper for `::fz_write_string()`.
    Write a string. Does not write zero terminator.
    """
    return _mupdf.ll_fz_write_string(out, s)