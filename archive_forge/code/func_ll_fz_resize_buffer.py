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
def ll_fz_resize_buffer(buf, capacity):
    """
    Low-level wrapper for `::fz_resize_buffer()`.
    Ensure that a buffer has a given capacity,
    truncating data if required.

    capacity: The desired capacity for the buffer. If the current
    size of the buffer contents is smaller than capacity, it is
    truncated.
    """
    return _mupdf.ll_fz_resize_buffer(buf, capacity)