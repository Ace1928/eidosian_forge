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
def ll_fz_open_leecher(chain, buf):
    """
    Low-level wrapper for `::fz_open_leecher()`.
    Attach a filter to a stream that will store any
    characters read from the stream into the supplied buffer.

    chain: The underlying stream to leech from.

    buf: The buffer into which the read data should be appended.
    The buffer will be resized as required.

    Returns pointer to newly created stream. May throw exceptions on
    failure to allocate.
    """
    return _mupdf.ll_fz_open_leecher(chain, buf)