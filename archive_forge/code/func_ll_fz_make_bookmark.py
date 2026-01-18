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
def ll_fz_make_bookmark(doc, loc):
    """
    Low-level wrapper for `::fz_make_bookmark()`.
    Create a bookmark for the given page, which can be used to find
    the same location after the document has been laid out with
    different parameters.
    """
    return _mupdf.ll_fz_make_bookmark(doc, loc)