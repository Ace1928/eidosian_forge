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
def ll_fz_malloc(size):
    """
    Low-level wrapper for `::fz_malloc()`.
    Allocate uninitialized memory of a given size.
    Does NOT clear the memory!

    May return NULL for size = 0.

    Throws exception in the event of failure to allocate.
    """
    return _mupdf.ll_fz_malloc(size)