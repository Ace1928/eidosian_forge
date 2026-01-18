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
def ll_fz_new_pdf_writer(path, options):
    """
    Low-level wrapper for `::fz_new_pdf_writer()`.
    Document writers for various possible output formats.

    All of the "_with_output" variants pass the ownership of out in
    immediately upon calling. The writers are responsible for
    dropping the fz_output when they are finished with it (even
    if they throw an exception during creation).
    """
    return _mupdf.ll_fz_new_pdf_writer(path, options)