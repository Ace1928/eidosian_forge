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
def ll_fz_invert_pixmap(pix):
    """
    Low-level wrapper for `::fz_invert_pixmap()`.
    Invert all the pixels in a pixmap. All components (process and
    spots) of all pixels are inverted (except alpha, which is
    unchanged).
    """
    return _mupdf.ll_fz_invert_pixmap(pix)