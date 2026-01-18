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
def ll_fz_write_rune(out, rune):
    """
    Low-level wrapper for `::fz_write_rune()`.
    Write a UTF-8 encoded unicode character.
    """
    return _mupdf.ll_fz_write_rune(out, rune)