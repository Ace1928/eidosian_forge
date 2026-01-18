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
def fz_slice_buffer(self, start, end):
    """
        Class-aware wrapper for `::fz_slice_buffer()`.
        	Create a new buffer with a (subset of) the data from the buffer.

        	start: if >= 0, offset from start of buffer, if < 0 offset from end of buffer.

        	end: if >= 0, offset from start of buffer, if < 0 offset from end of buffer.

        """
    return _mupdf.FzBuffer_fz_slice_buffer(self, start, end)