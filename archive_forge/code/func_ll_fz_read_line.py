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
def ll_fz_read_line(stm, buf, max):
    """
    Low-level wrapper for `::fz_read_line()`.
    Read a line from stream into the buffer until either a
    terminating newline or EOF, which it replaces with a null byte
    ('').

    Returns buf on success, and NULL when end of file occurs while
    no characters have been read.
    """
    return _mupdf.ll_fz_read_line(stm, buf, max)