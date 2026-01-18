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
def fz_cleanname(name):
    """
    Class-aware wrapper for `::fz_cleanname()`.
    	rewrite path to the shortest string that names the same path.

    	Eliminates multiple and trailing slashes, interprets "." and
    	"..". Overwrites the string in place.
    """
    return _mupdf.fz_cleanname(name)