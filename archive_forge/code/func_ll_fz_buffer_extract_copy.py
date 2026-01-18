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
def ll_fz_buffer_extract_copy(buffer):
    """
    Returns buffer data as a Python bytes instance, leaving the
    buffer unchanged.
    """
    assert isinstance(buffer, fz_buffer)
    return ll_fz_buffer_to_bytes_internal(buffer, clear=0)