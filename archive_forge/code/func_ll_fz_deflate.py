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
def ll_fz_deflate(dest, source, source_length, level):
    """
    Wrapper for out-params of fz_deflate().
    Returns: size_t compressed_length
    """
    outparams = ll_fz_deflate_outparams()
    ret = ll_fz_deflate_outparams_fn(dest, source, source_length, level, outparams)
    return outparams.compressed_length