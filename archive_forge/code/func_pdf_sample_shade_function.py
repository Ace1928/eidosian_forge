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
def pdf_sample_shade_function(shade, n, funcs, func, t0, t1):
    """
    Class-aware wrapper for `::pdf_sample_shade_function()`.

    This function has out-params. Python/C# wrappers look like:
    	`pdf_sample_shade_function(float shade[256][33], int n, int funcs, ::pdf_function **func, float t0, float t1)` =>
    """
    return _mupdf.pdf_sample_shade_function(shade, n, funcs, func, t0, t1)