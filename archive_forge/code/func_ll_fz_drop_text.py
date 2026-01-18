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
def ll_fz_drop_text(text):
    """
    Low-level wrapper for `::fz_drop_text()`.
    Decrement the reference count for the text object. When the
    reference count hits zero, the text object is freed.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_text(text)