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
def fz_document_open_fn_call(fn, stream, accel, dir):
    """
    Class-aware wrapper for `::fz_document_open_fn_call()`.   Helper for calling a `fz_document_open_fn` function pointer via Swig
    from Python/C#.
    """
    return _mupdf.fz_document_open_fn_call(fn, stream, accel, dir)