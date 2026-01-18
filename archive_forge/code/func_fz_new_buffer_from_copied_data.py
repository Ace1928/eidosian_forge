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
@staticmethod
def fz_new_buffer_from_copied_data(data, size):
    """
        Class-aware wrapper for `::fz_new_buffer_from_copied_data()`.
        	Create a new buffer containing a copy of the passed data.
        """
    return _mupdf.FzBuffer_fz_new_buffer_from_copied_data(data, size)