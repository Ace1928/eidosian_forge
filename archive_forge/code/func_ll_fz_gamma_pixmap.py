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
def ll_fz_gamma_pixmap(pix, gamma):
    """
    Low-level wrapper for `::fz_gamma_pixmap()`.
    Apply gamma correction to a pixmap. All components
    of all pixels are modified (except alpha, which is unchanged).

    gamma: The gamma value to apply; 1.0 for no change.
    """
    return _mupdf.ll_fz_gamma_pixmap(pix, gamma)