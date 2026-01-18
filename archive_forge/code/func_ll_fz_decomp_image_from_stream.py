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
def ll_fz_decomp_image_from_stream(stm, image, subarea, indexed, l2factor):
    """
    Wrapper for out-params of fz_decomp_image_from_stream().
    Returns: fz_pixmap *, int l2extra
    """
    outparams = ll_fz_decomp_image_from_stream_outparams()
    ret = ll_fz_decomp_image_from_stream_outparams_fn(stm, image, subarea, indexed, l2factor, outparams)
    return (ret, outparams.l2extra)