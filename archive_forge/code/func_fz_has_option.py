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
def fz_has_option(opts, key, val):
    """
    Class-aware wrapper for `::fz_has_option()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_has_option(const char *opts, const char *key)` => `(int, const char *val)`

    	Look for a given option (key) in the opts string. Return 1 if
    	it has it, and update *val to point to the value within opts.
    """
    return _mupdf.fz_has_option(opts, key, val)