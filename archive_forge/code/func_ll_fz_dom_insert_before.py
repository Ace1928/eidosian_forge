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
def ll_fz_dom_insert_before(node, new_elt):
    """
    Low-level wrapper for `::fz_dom_insert_before()`.
    Insert an element (new_elt), before another element (node),
    unlinking the new_elt from its current position if required.
    """
    return _mupdf.ll_fz_dom_insert_before(node, new_elt)