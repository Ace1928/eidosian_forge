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
def ll_fz_drop_stream(stm):
    """
    Low-level wrapper for `::fz_drop_stream()`.
    Decrements the reference count for a stream.

    When the reference count for the stream hits zero, frees the
    storage used for the fz_stream itself, and (usually)
    releases the underlying resources that the stream is based upon
    (depends on the method used to open the stream initially).
    """
    return _mupdf.ll_fz_drop_stream(stm)