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
def ll_fz_clone_stroke_state(stroke):
    """
    Low-level wrapper for `::fz_clone_stroke_state()`.
    Create an identical stroke_state structure and return a
    reference to it.

    stroke: The stroke state reference to clone.

    Exceptions may be thrown in the event of a failure to
    allocate.
    """
    return _mupdf.ll_fz_clone_stroke_state(stroke)