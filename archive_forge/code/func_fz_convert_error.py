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
def fz_convert_error(code):
    """
    Class-aware wrapper for `::fz_convert_error()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_convert_error()` => `(const char *, int code)`
    """
    return _mupdf.fz_convert_error(code)