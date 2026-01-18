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
def fz_rethrow_if(errcode):
    """
    Class-aware wrapper for `::fz_rethrow_if()`.
    	Within an fz_catch() block, rethrow the current exception
    	if the errcode of the current exception matches.

    	This assumes no intervening use of fz_try/fz_catch.
    """
    return _mupdf.fz_rethrow_if(errcode)