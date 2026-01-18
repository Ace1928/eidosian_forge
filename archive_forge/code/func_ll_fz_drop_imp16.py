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
def ll_fz_drop_imp16(p):
    """
    Wrapper for out-params of fz_drop_imp16().
    Returns: int, int16_t refs
    """
    outparams = ll_fz_drop_imp16_outparams()
    ret = ll_fz_drop_imp16_outparams_fn(p, outparams)
    return (ret, outparams.refs)