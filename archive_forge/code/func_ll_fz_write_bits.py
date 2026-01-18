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
def ll_fz_write_bits(out, data, num_bits):
    """
    Low-level wrapper for `::fz_write_bits()`.
    Write num_bits of data to the end of the output stream, assumed to be packed
    most significant bits first.
    """
    return _mupdf.ll_fz_write_bits(out, data, num_bits)