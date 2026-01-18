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
def ll_fz_arc4_encrypt(state, dest, src, len):
    """
    Low-level wrapper for `::fz_arc4_encrypt()`.
    RC4 block encrypt operation; encrypt src into dst (both of
    length len) updating the RC4 state as we go.

    Never throws an exception.
    """
    return _mupdf.ll_fz_arc4_encrypt(state, dest, src, len)