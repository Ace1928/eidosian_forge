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
def ll_fz_save_pixmap_as_pnm(pixmap, filename):
    """
    Low-level wrapper for `::fz_save_pixmap_as_pnm()`.
    Save a pixmap as a pnm (greyscale or rgb, no alpha).
    """
    return _mupdf.ll_fz_save_pixmap_as_pnm(pixmap, filename)