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
def ll_fz_new_display_list_from_page_contents(page):
    """
    Low-level wrapper for `::fz_new_display_list_from_page_contents()`.
    Create a display list from page contents (no annotations).

    Ownership of the display list is returned to the caller.
    """
    return _mupdf.ll_fz_new_display_list_from_page_contents(page)