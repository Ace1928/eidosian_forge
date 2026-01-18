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
def ll_fz_image_resolution(image):
    """
    Wrapper for out-params of fz_image_resolution().
    Returns: int xres, int yres
    """
    outparams = ll_fz_image_resolution_outparams()
    ret = ll_fz_image_resolution_outparams_fn(image, outparams)
    return (outparams.xres, outparams.yres)