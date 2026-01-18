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
def fz_fill_pixmap_with_color_outparams_fn(pix, colorspace, color_params):
    """
    Class-aware helper for out-params of fz_fill_pixmap_with_color() [fz_fill_pixmap_with_color()].
    """
    color = ll_fz_fill_pixmap_with_color(pix.m_internal, colorspace.m_internal, color_params.internal())
    return color