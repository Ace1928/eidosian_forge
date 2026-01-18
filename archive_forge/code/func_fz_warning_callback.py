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
def fz_warning_callback(user):
    """
    Class-aware wrapper for `::fz_warning_callback()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_warning_callback()` => `(fz_warning_cb *, void *user)`

    	Retrieve the currently set warning callback, or NULL if none
    	has been set. Optionally, if user is non-NULL, the user pointer
    	given when the warning callback was set is also passed back to
    	the caller.
    """
    return _mupdf.fz_warning_callback(user)