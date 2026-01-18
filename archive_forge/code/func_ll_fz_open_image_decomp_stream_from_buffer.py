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
def ll_fz_open_image_decomp_stream_from_buffer(arg_0):
    """
    Wrapper for out-params of fz_open_image_decomp_stream_from_buffer().
    Returns: fz_stream *, int l2factor
    """
    outparams = ll_fz_open_image_decomp_stream_from_buffer_outparams()
    ret = ll_fz_open_image_decomp_stream_from_buffer_outparams_fn(arg_0, outparams)
    return (ret, outparams.l2factor)