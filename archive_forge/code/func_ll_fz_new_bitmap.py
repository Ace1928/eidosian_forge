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
def ll_fz_new_bitmap(w, h, n, xres, yres):
    """
    Low-level wrapper for `::fz_new_bitmap()`.
    Create a new bitmap.

    w, h: Width and Height for the bitmap

    n: Number of color components (assumed to be a divisor of 8)

    xres, yres: X and Y resolutions (in pixels per inch).

    Returns pointer to created bitmap structure. The bitmap
    data is uninitialised.
    """
    return _mupdf.ll_fz_new_bitmap(w, h, n, xres, yres)