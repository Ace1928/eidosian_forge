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
def fz_strsep_outparams_fn(delim):
    """
    Class-aware helper for out-params of fz_strsep() [fz_strsep()].
    """
    ret, stringp = ll_fz_strsep(delim)
    return (ret, stringp)