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
def pdf_sample_shade_function_outparams_fn(shade, n, funcs, t0, t1):
    """
    Class-aware helper for out-params of pdf_sample_shade_function() [pdf_sample_shade_function()].
    """
    func = ll_pdf_sample_shade_function(shade, n, funcs, t0, t1)
    return PdfFunction(ll_pdf_keep_function(func))