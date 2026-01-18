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
def ll_fz_previous_page(doc, loc):
    """
    Low-level wrapper for `::fz_previous_page()`.
    Function to get the location of the previous page (allowing for
    the end of chapters etc). If already at the start of the
    document, returns the current page.
    """
    return _mupdf.ll_fz_previous_page(doc, loc)