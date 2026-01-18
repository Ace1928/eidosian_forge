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
def fz_tune_image_scale(image_scale, arg):
    """
    Class-aware wrapper for `::fz_tune_image_scale()`.
    	Set the tuning function to use for
    	image scaling.

    	image_scale: Function to use.

    	arg: Opaque argument to be passed to tuning function.
    """
    return _mupdf.fz_tune_image_scale(image_scale, arg)