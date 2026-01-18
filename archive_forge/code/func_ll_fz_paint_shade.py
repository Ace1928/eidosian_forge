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
def ll_fz_paint_shade(shade, override_cs, ctm, dest, color_params, bbox, eop):
    """
    Wrapper for out-params of fz_paint_shade().
    Returns: ::fz_shade_color_cache *cache
    """
    outparams = ll_fz_paint_shade_outparams()
    ret = ll_fz_paint_shade_outparams_fn(shade, override_cs, ctm, dest, color_params, bbox, eop, outparams)
    return outparams.cache