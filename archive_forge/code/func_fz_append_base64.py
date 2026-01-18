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
def fz_append_base64(self, data, size, newline):
    """
        Class-aware wrapper for `::fz_append_base64()`.
        	Write a base64 encoded data block, optionally with periodic newlines.
        """
    return _mupdf.FzBuffer_fz_append_base64(self, data, size, newline)