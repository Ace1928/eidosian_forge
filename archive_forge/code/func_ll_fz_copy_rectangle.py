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
def ll_fz_copy_rectangle(page, area, crlf):
    """
    Low-level wrapper for `::fz_copy_rectangle()`.
    Return a newly allocated UTF-8 string with the text for a given
    selection rectangle.

    crlf: If true, write "\\r\\n" style line endings (otherwise "\\n"
    only).
    """
    return _mupdf.ll_fz_copy_rectangle(page, area, crlf)