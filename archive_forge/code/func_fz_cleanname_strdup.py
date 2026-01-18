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
def fz_cleanname_strdup(name):
    """
    Class-aware wrapper for `::fz_cleanname_strdup()`.
    	rewrite path to the shortest string that names the same path.

    	Eliminates multiple and trailing slashes, interprets "." and
    	"..". Allocates a new string that the caller must free.
    """
    return _mupdf.fz_cleanname_strdup(name)