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
def ll_fz_write_pixmap_as_png(out, pixmap):
    """
    Low-level wrapper for `::fz_write_pixmap_as_png()`.
    Write a (Greyscale or RGB) pixmap as a png.
    """
    return _mupdf.ll_fz_write_pixmap_as_png(out, pixmap)