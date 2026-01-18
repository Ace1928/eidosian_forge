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
def ll_fz_display_list_is_empty(list):
    """
    Low-level wrapper for `::fz_display_list_is_empty()`.
    Check for a display list being empty

    list: The list to check.

    Returns true if empty, false otherwise.
    """
    return _mupdf.ll_fz_display_list_is_empty(list)