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
def ll_fz_pixmap_size(pix):
    """
    Low-level wrapper for `::fz_pixmap_size()`.
    Return sizeof fz_pixmap plus size of data, in bytes.
    """
    return _mupdf.ll_fz_pixmap_size(pix)