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
def ll_fz_font_digest(font, digest):
    """
    Low-level wrapper for `::fz_font_digest()`.
    Retrieve the MD5 digest for the font's data.
    """
    return _mupdf.ll_fz_font_digest(font, digest)