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
def ll_fz_drop_storable(arg_1):
    """
    Low-level wrapper for `::fz_drop_storable()`.
    Decrement the reference count for a storable object. When the
    reference count hits zero, the drop function for that object
    is called to free the object.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_storable(arg_1)