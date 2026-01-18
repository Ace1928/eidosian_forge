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
def ll_fz_is_valid_blend_colorspace(cs):
    """
    Low-level wrapper for `::fz_is_valid_blend_colorspace()`.
    Check to see that a colorspace is appropriate to be used as
    a blending space (i.e. only grey, rgb or cmyk).
    """
    return _mupdf.ll_fz_is_valid_blend_colorspace(cs)