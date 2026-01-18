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
def fz_decode_uri_component(s):
    """
     Class-aware wrapper for `::fz_decode_uri_component()`.
    Return a new string representing the unencoded version of the given URI component.
    This decodes all escape sequences!
    """
    return _mupdf.fz_decode_uri_component(s)