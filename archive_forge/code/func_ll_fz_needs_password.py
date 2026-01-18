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
def ll_fz_needs_password(doc):
    """
    Low-level wrapper for `::fz_needs_password()`.
    Check if a document is encrypted with a
    non-blank password.
    """
    return _mupdf.ll_fz_needs_password(doc)