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
def ll_fz_open_flated(chain, window_bits):
    """
    Low-level wrapper for `::fz_open_flated()`.
    flated filter performs LZ77 decoding (inflating) of data read
    from the chained filter.

    window_bits: How large a decompression window to use. Typically
    15. A negative number, -n, means to use n bits, but to expect
    raw data with no header.
    """
    return _mupdf.ll_fz_open_flated(chain, window_bits)