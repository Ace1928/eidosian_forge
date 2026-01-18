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
def ll_fz_getopt_long(nargc, ostr, longopts):
    """
    Wrapper for out-params of fz_getopt_long().
    Returns: int, char *nargv
    """
    outparams = ll_fz_getopt_long_outparams()
    ret = ll_fz_getopt_long_outparams_fn(nargc, ostr, longopts, outparams)
    return (ret, outparams.nargv)