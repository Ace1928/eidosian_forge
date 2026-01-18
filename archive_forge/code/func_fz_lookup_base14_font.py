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
def fz_lookup_base14_font(name, len):
    """
    Class-aware wrapper for `::fz_lookup_base14_font()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_lookup_base14_font(const char *name)` => `(const unsigned char *, int len)`

    	Search the builtin base14 fonts for a match.
    	Whether a given font is present or not will depend on the
    	configuration in which MuPDF is built.

    	name: The name of the font desired.

    	len: Pointer to a place to receive the length of the discovered
    	font buffer.

    	Returns a pointer to the font file data, or NULL if not present.
    """
    return _mupdf.fz_lookup_base14_font(name, len)