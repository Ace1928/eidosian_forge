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
def ll_fz_read_byte(stm):
    """
    Low-level wrapper for `::fz_read_byte()`.
    Read the next byte from a stream.

    stm: The stream t read from.

    Returns -1 for end of stream, or the next byte. May
    throw exceptions.
    """
    return _mupdf.ll_fz_read_byte(stm)