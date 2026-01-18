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
def ll_fz_dom_previous(elt):
    """
    Low-level wrapper for `::fz_dom_previous()`.
    Return a borrowed reference to the previous sibling of a node,
    or NULL if there isn't one.
    """
    return _mupdf.ll_fz_dom_previous(elt)