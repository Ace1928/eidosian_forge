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
def ll_fz_is_document_reflowable(doc):
    """
    Low-level wrapper for `::fz_is_document_reflowable()`.
    Is the document reflowable.

    Returns 1 to indicate reflowable documents, otherwise 0.
    """
    return _mupdf.ll_fz_is_document_reflowable(doc)