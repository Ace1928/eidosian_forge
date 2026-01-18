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
def ll_fz_peek_byte(stm):
    """
    Low-level wrapper for `::fz_peek_byte()`.
    Peek at the next byte in a stream.

    stm: The stream to peek at.

    Returns -1 for EOF, or the next byte that will be read.
    """
    return _mupdf.ll_fz_peek_byte(stm)