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
def ll_fz_new_buffer_from_data(data, size):
    """
    Low-level wrapper for `::fz_new_buffer_from_data()`.
    Create a new buffer with existing data.

    data: Pointer to existing data.
    size: Size of existing data.

    Takes ownership of data. Does not make a copy. Calls fz_free on
    the data when the buffer is deallocated. Do not use 'data' after
    passing to this function.

    Returns pointer to new buffer. Throws exception on allocation
    failure.
    """
    return _mupdf.ll_fz_new_buffer_from_data(data, size)