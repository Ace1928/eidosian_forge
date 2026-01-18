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
def fz_grisu(f, s, exp):
    """
    Class-aware wrapper for `::fz_grisu()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_grisu(float f, char *s)` => `(int, int exp)`
    """
    return _mupdf.fz_grisu(f, s, exp)