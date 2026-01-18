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
def ll_fz_drop_document(doc):
    """
    Low-level wrapper for `::fz_drop_document()`.
    Decrement the document reference count. When the reference
    count reaches 0, the document and all it's references are
    freed.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_document(doc)