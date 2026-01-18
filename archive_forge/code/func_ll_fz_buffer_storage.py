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
def ll_fz_buffer_storage(buf):
    """
    Wrapper for out-params of fz_buffer_storage().
    Returns: size_t, unsigned char *datap
    """
    outparams = ll_fz_buffer_storage_outparams()
    ret = ll_fz_buffer_storage_outparams_fn(buf, outparams)
    return (ret, outparams.datap)