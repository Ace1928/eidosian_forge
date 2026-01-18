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
def ll_fz_font_t3_procs(font):
    """
    Low-level wrapper for `::fz_font_t3_procs()`.
    Retrieve the Type3 procs
    for a font.

    font: The font to query

    Returns the t3_procs pointer. Will be NULL for a
    non type-3 font.
    """
    return _mupdf.ll_fz_font_t3_procs(font)