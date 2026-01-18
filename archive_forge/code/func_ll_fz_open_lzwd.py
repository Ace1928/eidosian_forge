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
def ll_fz_open_lzwd(chain, early_change, min_bits, reverse_bits, old_tiff):
    """
    Low-level wrapper for `::fz_open_lzwd()`.
    lzwd filter performs LZW decoding of data read from the chained
    filter.

    early_change: (Default 1) specifies whether to change codes 1
    bit early.

    min_bits: (Default 9) specifies the minimum number of bits to
    use.

    reverse_bits: (Default 0) allows for compatibility with gif and
    old style tiffs (1).

    old_tiff: (Default 0) allows for different handling of the clear
    code, as found in old style tiffs.
    """
    return _mupdf.ll_fz_open_lzwd(chain, early_change, min_bits, reverse_bits, old_tiff)