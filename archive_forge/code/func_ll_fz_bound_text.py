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
def ll_fz_bound_text(text, stroke, ctm):
    """
    Low-level wrapper for `::fz_bound_text()`.
    Find the bounds of a given text object.

    text: The text object to find the bounds of.

    stroke: Pointer to the stroke attributes (for stroked
    text), or NULL (for filled text).

    ctm: The matrix in use.

    r: pointer to storage for the bounds.

    Returns a pointer to r, which is updated to contain the
    bounding box for the text object.
    """
    return _mupdf.ll_fz_bound_text(text, stroke, ctm)