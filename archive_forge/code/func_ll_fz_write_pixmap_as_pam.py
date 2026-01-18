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
def ll_fz_write_pixmap_as_pam(out, pixmap):
    """
    Low-level wrapper for `::fz_write_pixmap_as_pam()`.
    Write a pixmap as a pnm (greyscale, rgb or cmyk, with or without
    alpha).
    """
    return _mupdf.ll_fz_write_pixmap_as_pam(out, pixmap)