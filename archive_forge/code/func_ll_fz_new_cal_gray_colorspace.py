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
def ll_fz_new_cal_gray_colorspace(wp, bp, gamma):
    """
    Low-level wrapper for `::fz_new_cal_gray_colorspace()`.
    Create a calibrated gray colorspace.

    The returned reference should be dropped when it is finished
    with.

    Colorspaces are immutable once created.
    """
    return _mupdf.ll_fz_new_cal_gray_colorspace(wp, bp, gamma)