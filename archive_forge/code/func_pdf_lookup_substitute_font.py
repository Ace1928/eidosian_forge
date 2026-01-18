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
def pdf_lookup_substitute_font(mono, serif, bold, italic, len):
    """
    Class-aware wrapper for `::pdf_lookup_substitute_font()`.

    This function has out-params. Python/C# wrappers look like:
    	`pdf_lookup_substitute_font(int mono, int serif, int bold, int italic)` => `(const unsigned char *, int len)`
    """
    return _mupdf.pdf_lookup_substitute_font(mono, serif, bold, italic, len)