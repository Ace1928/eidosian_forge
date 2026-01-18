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
def ll_fz_tint_pixmap(pix, black, white):
    """
    Low-level wrapper for `::fz_tint_pixmap()`.
    Tint all the pixels in an RGB, BGR, or Gray pixmap.

    black: Map black to this hexadecimal RGB color.

    white: Map white to this hexadecimal RGB color.
    """
    return _mupdf.ll_fz_tint_pixmap(pix, black, white)