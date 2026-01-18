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
def fz_strtof(s, es):
    """
    Class-aware wrapper for `::fz_strtof()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_strtof(const char *s)` => `(float, char *es)`

    	Locale-independent decimal to binary conversion. On overflow
    	return (-)INFINITY and set errno to ERANGE. On underflow return
    	0 and set errno to ERANGE. Special inputs (case insensitive):
    	"NAN", "INF" or "INFINITY".
    """
    return _mupdf.fz_strtof(s, es)