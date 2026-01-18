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
def ll_fz_has_option(opts, key):
    """
    Wrapper for out-params of fz_has_option().
    Returns: int, const char *val
    """
    outparams = ll_fz_has_option_outparams()
    ret = ll_fz_has_option_outparams_fn(opts, key, outparams)
    return (ret, outparams.val)