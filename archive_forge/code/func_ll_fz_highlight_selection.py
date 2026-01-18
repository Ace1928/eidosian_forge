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
def ll_fz_highlight_selection(page, a, b, quads, max_quads):
    """
    Low-level wrapper for `::fz_highlight_selection()`.
    Return a list of quads to highlight lines inside the selection
    points.
    """
    return _mupdf.ll_fz_highlight_selection(page, a, b, quads, max_quads)