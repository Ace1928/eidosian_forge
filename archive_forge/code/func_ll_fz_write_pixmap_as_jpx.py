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
def ll_fz_write_pixmap_as_jpx(out, pix, quality):
    """
    Low-level wrapper for `::fz_write_pixmap_as_jpx()`.
    Pixmap data as JP2K with no subsampling.

    quality = 100 = lossless
    otherwise for a factor of x compression use 100-x. (so 80 is 1:20 compression)
    """
    return _mupdf.ll_fz_write_pixmap_as_jpx(out, pix, quality)