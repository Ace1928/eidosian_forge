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
def ll_fz_pixmap_samples(pix):
    """
    Low-level wrapper for `::fz_pixmap_samples()`.
    Returns a pointer to the pixel data of a pixmap.

    Returns the pointer.
    """
    return _mupdf.ll_fz_pixmap_samples(pix)