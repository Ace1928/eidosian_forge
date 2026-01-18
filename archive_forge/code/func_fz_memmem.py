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
def fz_memmem(haystack, haystacklen, needle, needlelen):
    """
    Class-aware wrapper for `::fz_memmem()`.
    	Find the start of the first occurrence of the substring needle in haystack.
    """
    return _mupdf.fz_memmem(haystack, haystacklen, needle, needlelen)