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
def ll_fz_drop_store_context():
    """
    Low-level wrapper for `::fz_drop_store_context()`.
    Decrement the reference count for the store context. When the
    reference count hits zero, the store context is freed.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_store_context()