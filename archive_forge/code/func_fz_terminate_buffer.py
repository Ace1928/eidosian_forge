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
def fz_terminate_buffer(self):
    """
        Class-aware wrapper for `::fz_terminate_buffer()`.
        	Zero-terminate buffer in order to use as a C string.

        	This byte is invisible and does not affect the length of the
        	buffer as returned by fz_buffer_storage. The zero byte is
        	written *after* the data, and subsequent writes will overwrite
        	the terminating byte.

        	Subsequent changes to the size of the buffer (such as by
        	fz_buffer_trim, fz_buffer_grow, fz_resize_buffer, etc) may
        	invalidate this.
        """
    return _mupdf.FzBuffer_fz_terminate_buffer(self)