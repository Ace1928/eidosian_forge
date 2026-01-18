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
def ll_fz_read_rune(_in):
    """
    Low-level wrapper for `::fz_read_rune()`.
    Read a utf-8 rune from a stream.

    In the event of encountering badly formatted utf-8 codes
    (such as a leading code with an unexpected number of following
    codes) no error/exception is given, but undefined values may be
    returned.
    """
    return _mupdf.ll_fz_read_rune(_in)