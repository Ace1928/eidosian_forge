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
def fz_chartorune(rune, str):
    """
    Class-aware wrapper for `::fz_chartorune()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_chartorune(const char *str)` => `(int, int rune)`

    	UTF8 decode a single rune from a sequence of chars.

    	rune: Pointer to an int to assign the decoded 'rune' to.

    	str: Pointer to a UTF8 encoded string.

    	Returns the number of bytes consumed.
    """
    return _mupdf.fz_chartorune(rune, str)