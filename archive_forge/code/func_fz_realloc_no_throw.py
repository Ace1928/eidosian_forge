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
def fz_realloc_no_throw(p, size):
    """
    Class-aware wrapper for `::fz_realloc_no_throw()`.
    	fz_realloc equivalent that returns NULL rather than throwing
    	exceptions.
    """
    return _mupdf.fz_realloc_no_throw(p, size)