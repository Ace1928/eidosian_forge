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
def ll_fz_read_bits(stm, n):
    """
    Low-level wrapper for `::fz_read_bits()`.
    Read the next n bits from a stream (assumed to
    be packed most significant bit first).

    stm: The stream to read from.

    n: The number of bits to read, between 1 and 8*sizeof(int)
    inclusive.

    Returns -1 for EOF, or the required number of bits.
    """
    return _mupdf.ll_fz_read_bits(stm, n)