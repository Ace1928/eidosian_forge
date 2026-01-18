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
def fz_strlcat(dst, src, n):
    """
    Class-aware wrapper for `::fz_strlcat()`.
    	Concatenate 2 strings, with a maximum length.

    	dst: pointer to first string in a buffer of n bytes.

    	src: pointer to string to concatenate.

    	n: Size (in bytes) of buffer that dst is in.

    	Returns the real length that a concatenated dst + src would have
    	been (not including terminator).
    """
    return _mupdf.fz_strlcat(dst, src, n)