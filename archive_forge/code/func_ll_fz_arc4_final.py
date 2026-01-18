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
def ll_fz_arc4_final(state):
    """
    Low-level wrapper for `::fz_arc4_final()`.
    RC4 finalization. Zero the context.

    Never throws an exception.
    """
    return _mupdf.ll_fz_arc4_final(state)