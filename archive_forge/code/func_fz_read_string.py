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
def fz_read_string(self, buffer, len):
    """
        Class-aware wrapper for `::fz_read_string()`.
        	Read a null terminated string from the stream into
        	a buffer of a given length. The buffer will be null terminated.
        	Throws on failure (including the failure to fit the entire
        	string including the terminator into the buffer).
        """
    return _mupdf.FzStream_fz_read_string(self, buffer, len)