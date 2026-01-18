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
def ll_fz_dom_attribute(elt, att):
    """
    Low-level wrapper for `::fz_dom_attribute()`.
    Retrieve the value of a given attribute from a given element.

    Returns a borrowed pointer to the value or NULL if not found.
    """
    return _mupdf.ll_fz_dom_attribute(elt, att)