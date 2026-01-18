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
def ll_fz_strlcpy(dst, src, n):
    """
    Low-level wrapper for `::fz_strlcpy()`.
    Copy at most n-1 chars of a string into a destination
    buffer with null termination, returning the real length of the
    initial string (excluding terminator).

    dst: Destination buffer, at least n bytes long.

    src: C string (non-NULL).

    n: Size of dst buffer in bytes.

    Returns the length (excluding terminator) of src.
    """
    return _mupdf.ll_fz_strlcpy(dst, src, n)