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
def ll_fz_new_log_for_module(module):
    """
    Low-level wrapper for `::fz_new_log_for_module()`.
    Internal function to actually do the opening of the logfile.

    Caller should close/drop the output when finished with it.
    """
    return _mupdf.ll_fz_new_log_for_module(module)