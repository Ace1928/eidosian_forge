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
def ll_fz_keep_halftone(half):
    """
    Low-level wrapper for `::fz_keep_halftone()`.
    Take an additional reference to the halftone. The same pointer
    is returned.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_keep_halftone(half)