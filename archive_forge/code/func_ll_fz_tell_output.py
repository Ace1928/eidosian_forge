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
def ll_fz_tell_output(out):
    """
    Low-level wrapper for `::fz_tell_output()`.
    Return the current file position.

    Throw an error on untellable outputs.
    """
    return _mupdf.ll_fz_tell_output(out)