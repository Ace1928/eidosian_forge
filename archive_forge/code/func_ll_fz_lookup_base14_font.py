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
def ll_fz_lookup_base14_font(name):
    """
    Wrapper for out-params of fz_lookup_base14_font().
    Returns: const unsigned char *, int len
    """
    outparams = ll_fz_lookup_base14_font_outparams()
    ret = ll_fz_lookup_base14_font_outparams_fn(name, outparams)
    return (ret, outparams.len)