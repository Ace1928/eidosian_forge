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
def ll_fz_unread_byte(stm):
    """
    Low-level wrapper for `::fz_unread_byte()`.
    Unread the single last byte successfully
    read from a stream. Do not call this without having
    successfully read a byte.

    stm: The stream to operate upon.
    """
    return _mupdf.ll_fz_unread_byte(stm)