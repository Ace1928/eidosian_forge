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
def ll_fz_write_pwg_file_header(out):
    """
    Low-level wrapper for `::fz_write_pwg_file_header()`.
    Output the file header to a pwg stream, ready for pages to follow it.
    """
    return _mupdf.ll_fz_write_pwg_file_header(out)