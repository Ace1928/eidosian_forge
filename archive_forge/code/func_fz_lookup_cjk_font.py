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
def fz_lookup_cjk_font(ordering, len, index):
    """
    Class-aware wrapper for `::fz_lookup_cjk_font()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_lookup_cjk_font(int ordering)` => `(const unsigned char *, int len, int index)`

    	Search the builtin cjk fonts for a match.
    	Whether a font is present or not will depend on the
    	configuration in which MuPDF is built.

    	ordering: The desired ordering of the font (e.g. FZ_ADOBE_KOREA).

    	len: Pointer to a place to receive the length of the discovered
    	font buffer.

    	Returns a pointer to the font file data, or NULL if not present.
    """
    return _mupdf.fz_lookup_cjk_font(ordering, len, index)