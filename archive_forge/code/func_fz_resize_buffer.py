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
def fz_resize_buffer(self, capacity):
    """
        Class-aware wrapper for `::fz_resize_buffer()`.
        	Ensure that a buffer has a given capacity,
        	truncating data if required.

        	capacity: The desired capacity for the buffer. If the current
        	size of the buffer contents is smaller than capacity, it is
        	truncated.
        """
    return _mupdf.FzBuffer_fz_resize_buffer(self, capacity)