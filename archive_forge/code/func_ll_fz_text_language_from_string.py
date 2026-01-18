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
def ll_fz_text_language_from_string(str):
    """
    Low-level wrapper for `::fz_text_language_from_string()`.
    Convert ISO 639 (639-{1,2,3,5}) language specification
    strings losslessly to a 15 bit fz_text_language code.

    No validation is carried out. Obviously invalid (out
    of spec) codes will be mapped to FZ_LANG_UNSET, but
    well-formed (but undefined) codes will be blithely
    accepted.
    """
    return _mupdf.ll_fz_text_language_from_string(str)