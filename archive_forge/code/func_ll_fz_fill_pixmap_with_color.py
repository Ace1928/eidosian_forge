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
def ll_fz_fill_pixmap_with_color(pix, colorspace, color_params):
    """
    Wrapper for out-params of fz_fill_pixmap_with_color().
    Returns: float color
    """
    outparams = ll_fz_fill_pixmap_with_color_outparams()
    ret = ll_fz_fill_pixmap_with_color_outparams_fn(pix, colorspace, color_params, outparams)
    return outparams.color