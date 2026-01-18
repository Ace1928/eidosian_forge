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
def fz_image_resolution_outparams_fn(image):
    """
    Class-aware helper for out-params of fz_image_resolution() [fz_image_resolution()].
    """
    xres, yres = ll_fz_image_resolution(image.m_internal)
    return (xres, yres)