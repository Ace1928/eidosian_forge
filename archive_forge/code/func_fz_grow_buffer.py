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
def fz_grow_buffer(self):
    """
        Class-aware wrapper for `::fz_grow_buffer()`.
        	Make some space within a buffer (i.e. ensure that
        	capacity > size).
        """
    return _mupdf.FzBuffer_fz_grow_buffer(self)