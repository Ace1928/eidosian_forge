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
def ll_fz_new_output_with_buffer(buf):
    """
    Low-level wrapper for `::fz_new_output_with_buffer()`.
    Open an output stream that appends
    to a buffer.

    buf: The buffer to append to.
    """
    return _mupdf.ll_fz_new_output_with_buffer(buf)