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
def ll_fz_encode_uri_component(s):
    """
     Low-level wrapper for `::fz_encode_uri_component()`.
    Return a new string representing the provided string encoded as an URI component.
    This also encodes the special reserved characters (; / ? : @ & = + $ , #).
    """
    return _mupdf.ll_fz_encode_uri_component(s)