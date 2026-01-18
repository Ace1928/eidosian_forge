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
def ll_fz_free(p):
    """
    Low-level wrapper for `::fz_free()`.
    Free a previously allocated block of memory.

    fz_free(ctx, NULL) does nothing.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_free(p)