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
def ll_fz_keep_stroke_state(stroke):
    """
    Low-level wrapper for `::fz_keep_stroke_state()`.
    Take an additional reference to a stroke state structure.

    No modifications should be carried out on a stroke
    state to which more than one reference is held, as
    this can cause race conditions.
    """
    return _mupdf.ll_fz_keep_stroke_state(stroke)