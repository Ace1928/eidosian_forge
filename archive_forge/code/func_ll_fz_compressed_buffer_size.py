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
def ll_fz_compressed_buffer_size(buffer):
    """
    Low-level wrapper for `::fz_compressed_buffer_size()`.
    Return the storage size used for a buffer and its data.
    Used in implementing store handling.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_compressed_buffer_size(buffer)