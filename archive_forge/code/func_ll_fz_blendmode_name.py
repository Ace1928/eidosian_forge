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
def ll_fz_blendmode_name(blendmode):
    """
    Low-level wrapper for `::fz_blendmode_name()`.
    Map from enumeration to blend mode string.

    The string is static, with arbitrary lifespan.
    """
    return _mupdf.ll_fz_blendmode_name(blendmode)