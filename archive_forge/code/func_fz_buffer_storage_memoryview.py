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
def fz_buffer_storage_memoryview(buffer, writable=False):
    """
    Returns a read-only or writable Python `memoryview` onto
    `fz_buffer` data. This relies on `buffer` existing and
    not changing size while the `memoryview` is used.
    """
    assert isinstance(buffer, FzBuffer)
    return ll_fz_buffer_storage_memoryview(buffer.m_internal, writable)