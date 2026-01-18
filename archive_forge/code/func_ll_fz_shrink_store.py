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
def ll_fz_shrink_store(percent):
    """
    Low-level wrapper for `::fz_shrink_store()`.
    Evict items from the store until the total size of
    the objects in the store is reduced to a given percentage of its
    current size.

    percent: %age of current size to reduce the store to.

    Returns non zero if we managed to free enough memory, zero
    otherwise.
    """
    return _mupdf.ll_fz_shrink_store(percent)