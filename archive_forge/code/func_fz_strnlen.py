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
def fz_strnlen(s, maxlen):
    """
    Class-aware wrapper for `::fz_strnlen()`.
    	Return strlen(s), if that is less than maxlen, or maxlen if
    	there is no null byte ('') among the first maxlen bytes.
    """
    return _mupdf.fz_strnlen(s, maxlen)