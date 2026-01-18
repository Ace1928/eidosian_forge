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
def fz_default_warning_callback(user, message):
    """
    Class-aware wrapper for `::fz_default_warning_callback()`.
    	The default warning callback. Declared publicly just so that
    	the warning callback can be set back to this after it has been
    	overridden.
    """
    return _mupdf.fz_default_warning_callback(user, message)