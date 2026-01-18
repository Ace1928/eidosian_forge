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
def ll_fz_try_read_file(filename):
    """
    Low-level wrapper for `::fz_try_read_file()`.
    Read all the contents of a file into a buffer.

    Returns NULL if the file does not exist, otherwise
    behaves exactly as fz_read_file.
    """
    return _mupdf.ll_fz_try_read_file(filename)