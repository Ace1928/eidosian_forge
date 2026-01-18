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
def ll_fz_drop_path(path):
    """
    Low-level wrapper for `::fz_drop_path()`.
    Decrement the reference count. When the reference count hits
    zero, free the path.

    All paths can be dropped, regardless of their packing type.
    Packed paths do not own the blocks into which they are packed
    so dropping them does not free those blocks.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_path(path)