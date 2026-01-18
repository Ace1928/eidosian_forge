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
def ll_fz_new_buffer_from_shared_data(data, size):
    """
    Low-level wrapper for `::fz_new_buffer_from_shared_data()`.
    Like fz_new_buffer, but does not take ownership.
    """
    return _mupdf.ll_fz_new_buffer_from_shared_data(data, size)