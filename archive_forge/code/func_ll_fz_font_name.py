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
def ll_fz_font_name(font):
    """
    Low-level wrapper for `::fz_font_name()`.
    Retrieve a pointer to the name of the font.

    font: The font to query.

    Returns a pointer to an internal copy of the font name.
    Will never be NULL, but may be the empty string.
    """
    return _mupdf.ll_fz_font_name(font)