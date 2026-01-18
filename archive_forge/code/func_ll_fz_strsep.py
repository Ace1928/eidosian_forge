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
def ll_fz_strsep(delim):
    """
    Wrapper for out-params of fz_strsep().
    Returns: char *, char *stringp
    """
    outparams = ll_fz_strsep_outparams()
    ret = ll_fz_strsep_outparams_fn(delim, outparams)
    return (ret, outparams.stringp)