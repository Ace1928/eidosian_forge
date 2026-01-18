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
def ll_fz_new_output(bufsiz, state, write, close, drop):
    """
    Low-level wrapper for `::fz_new_output()`.
    Create a new output object with the given
    internal state and function pointers.

    state: Internal state (opaque to everything but implementation).

    write: Function to output a given buffer.

    close: Cleanup function to destroy state when output closed.
    May permissibly be null.
    """
    return _mupdf.ll_fz_new_output(bufsiz, state, write, close, drop)