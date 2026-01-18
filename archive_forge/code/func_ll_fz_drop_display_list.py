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
def ll_fz_drop_display_list(list):
    """
    Low-level wrapper for `::fz_drop_display_list()`.
    Decrement the reference count for a display list. When the
    reference count reaches zero, all the references in the display
    list itself are dropped, and the display list is freed.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_display_list(list)