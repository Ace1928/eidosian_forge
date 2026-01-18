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
def fz_default_error_callback(user, message):
    """
    Class-aware wrapper for `::fz_default_error_callback()`.
    	FIXME: Better not to expose fz_default_error_callback, and
    	fz_default_warning callback and to allow 'NULL' to be used
    	int fz_set_xxxx_callback to mean "defaults".

    	FIXME: Do we need/want functions like
    	fz_error_callback(ctx, message) to allow callers to inject
    	stuff into the error/warning streams?

    	The default error callback. Declared publicly just so that the
    	error callback can be set back to this after it has been
    	overridden.
    """
    return _mupdf.fz_default_error_callback(user, message)