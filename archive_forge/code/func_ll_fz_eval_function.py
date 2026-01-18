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
def ll_fz_eval_function(func, in_, inlen, outlen):
    """
    Wrapper for out-params of fz_eval_function().
    Returns: float out
    """
    outparams = ll_fz_eval_function_outparams()
    ret = ll_fz_eval_function_outparams_fn(func, in_, inlen, outlen, outparams)
    return outparams.out