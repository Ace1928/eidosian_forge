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
def ll_fz_open_accelerated_document(filename, accel):
    """
    Low-level wrapper for `::fz_open_accelerated_document()`.
    Open a document file and read its basic structure so pages and
    objects can be located. MuPDF will try to repair broken
    documents (without actually changing the file contents).

    The returned fz_document is used when calling most other
    document related functions.

    filename: a path to a file as it would be given to open(2).
    """
    return _mupdf.ll_fz_open_accelerated_document(filename, accel)