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
def fz_getopt_long_outparams_fn(nargc, ostr, longopts):
    """
    Class-aware helper for out-params of fz_getopt_long() [fz_getopt_long()].
    """
    ret, nargv = ll_fz_getopt_long(nargc, ostr, longopts.m_internal)
    return (ret, nargv)