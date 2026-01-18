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
def ll_fz_drop_bitmap(bit):
    """
    Low-level wrapper for `::fz_drop_bitmap()`.
    Drop a reference to the bitmap. When the reference count reaches
    zero, the bitmap will be destroyed.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_bitmap(bit)