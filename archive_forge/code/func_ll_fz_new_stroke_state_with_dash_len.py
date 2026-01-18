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
def ll_fz_new_stroke_state_with_dash_len(len):
    """
    Low-level wrapper for `::fz_new_stroke_state_with_dash_len()`.
    Create a new (empty) stroke state structure, with room for
    dash data of the given length, and return a reference to it.

    len: The number of dash elements to allow room for.

    Throws exception on failure to allocate.
    """
    return _mupdf.ll_fz_new_stroke_state_with_dash_len(len)