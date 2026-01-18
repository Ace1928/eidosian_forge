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
def ll_fz_stderr():
    """
    Low-level wrapper for `::fz_stderr()`.
    Retrieve an fz_output that directs to stdout.

    Optionally may be fz_dropped when finished with.
    """
    return _mupdf.ll_fz_stderr()