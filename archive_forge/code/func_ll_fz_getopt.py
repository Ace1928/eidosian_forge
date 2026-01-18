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
def ll_fz_getopt(nargc, ostr):
    """
    Wrapper for out-params of fz_getopt().
    Returns: int, char *nargv
    """
    outparams = ll_fz_getopt_outparams()
    ret = ll_fz_getopt_outparams_fn(nargc, ostr, outparams)
    return (ret, outparams.nargv)