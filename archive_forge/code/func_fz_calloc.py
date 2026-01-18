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
def fz_calloc(count, size):
    """
    Class-aware wrapper for `::fz_calloc()`.
    	Allocate array of memory of count entries of size bytes.
    	Clears the memory to zero.

    	Throws exception in the event of failure to allocate.
    """
    return _mupdf.fz_calloc(count, size)