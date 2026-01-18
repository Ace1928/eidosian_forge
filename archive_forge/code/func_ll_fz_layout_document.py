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
def ll_fz_layout_document(doc, w, h, em):
    """
    Low-level wrapper for `::fz_layout_document()`.
    Layout reflowable document types.

    w, h: Page size in points.
    em: Default font size in points.
    """
    return _mupdf.ll_fz_layout_document(doc, w, h, em)