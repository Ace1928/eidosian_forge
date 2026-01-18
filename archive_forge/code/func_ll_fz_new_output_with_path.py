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
def ll_fz_new_output_with_path(filename, append):
    """
    Low-level wrapper for `::fz_new_output_with_path()`.
    Open an output stream that writes to a
    given path.

    filename: The filename to write to (specified in UTF-8).

    append: non-zero if we should append to the file, rather than
    overwriting it.
    """
    return _mupdf.ll_fz_new_output_with_path(filename, append)