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
def ll_fz_new_document_writer_with_output(out, format, options):
    """
    Low-level wrapper for `::fz_new_document_writer_with_output()`.
    Like fz_new_document_writer but takes a fz_output for writing
    the result. Only works for multi-page formats.
    """
    return _mupdf.ll_fz_new_document_writer_with_output(out, format, options)