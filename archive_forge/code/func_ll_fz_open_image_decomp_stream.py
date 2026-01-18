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
def ll_fz_open_image_decomp_stream(arg_0, arg_1):
    """
    Wrapper for out-params of fz_open_image_decomp_stream().
    Returns: fz_stream *, int l2factor
    """
    outparams = ll_fz_open_image_decomp_stream_outparams()
    ret = ll_fz_open_image_decomp_stream_outparams_fn(arg_0, arg_1, outparams)
    return (ret, outparams.l2factor)