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
def ll_pdf_sample_shade_function(shade, n, funcs, t0, t1):
    """
    Wrapper for out-params of pdf_sample_shade_function().
    Returns: ::pdf_function *func
    """
    outparams = ll_pdf_sample_shade_function_outparams()
    ret = ll_pdf_sample_shade_function_outparams_fn(shade, n, funcs, t0, t1, outparams)
    return outparams.func