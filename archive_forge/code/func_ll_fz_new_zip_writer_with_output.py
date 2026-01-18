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
def ll_fz_new_zip_writer_with_output(out):
    """
    Low-level wrapper for `::fz_new_zip_writer_with_output()`.
    Create a new zip writer that writes to a given output stream.

    Ownership of out passes in immediately upon calling this function.
    The caller should never drop the fz_output, even if this function throws
    an exception.
    """
    return _mupdf.ll_fz_new_zip_writer_with_output(out)