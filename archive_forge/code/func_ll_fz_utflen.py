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
def ll_fz_utflen(s):
    """
    Low-level wrapper for `::fz_utflen()`.
    Count how many runes the UTF-8 encoded string
    consists of.

    s: The UTF-8 encoded, NUL-terminated text string.

    Returns the number of runes in the string.
    """
    return _mupdf.ll_fz_utflen(s)