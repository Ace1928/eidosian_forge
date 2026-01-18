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
def ll_fz_invert_pixmap_luminance(pix):
    """
    Low-level wrapper for `::fz_invert_pixmap_luminance()`.
    Transform the pixels in a pixmap so that luminance of each
    pixel is inverted, and the chrominance remains unchanged (as
    much as accuracy allows).

    All components of all pixels are inverted (except alpha, which
    is unchanged). Only supports Grey and RGB bitmaps.
    """
    return _mupdf.ll_fz_invert_pixmap_luminance(pix)