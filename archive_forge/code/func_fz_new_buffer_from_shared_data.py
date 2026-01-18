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
def fz_new_buffer_from_shared_data(data, size):
    """
    Class-aware wrapper for `::fz_new_buffer_from_shared_data()`.
    	Like fz_new_buffer, but does not take ownership.
    """
    return _mupdf.fz_new_buffer_from_shared_data(data, size)