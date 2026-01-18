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
def ll_fz_drop_context():
    """
    Low-level wrapper for `::fz_drop_context()`.
    Free a context and its global state.

    The context and all of its global state is freed, and any
    buffered warnings are flushed (see fz_flush_warnings). If NULL
    is passed in nothing will happen.

    Must not be called for a context that is being used in an active
    fz_try(), fz_always() or fz_catch() block.
    """
    return _mupdf.ll_fz_drop_context()