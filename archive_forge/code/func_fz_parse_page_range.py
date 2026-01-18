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
def fz_parse_page_range(s, a, b, n):
    """
    Class-aware wrapper for `::fz_parse_page_range()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_parse_page_range(const char *s, int n)` => `(const char *, int a, int b)`
    """
    return _mupdf.fz_parse_page_range(s, a, b, n)