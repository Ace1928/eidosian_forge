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
def ll_fz_recognize_document(magic):
    """
    Low-level wrapper for `::fz_recognize_document()`.
    Given a magic find a document handler that can handle a
    document of this type.

    magic: Can be a filename extension (including initial period) or
    a mimetype.
    """
    return _mupdf.ll_fz_recognize_document(magic)