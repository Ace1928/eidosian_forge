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
def ll_fz_open_endstream_filter(chain, len, offset):
    """
    Low-level wrapper for `::fz_open_endstream_filter()`.
    The endstream filter reads a PDF substream, and starts to look
    for an 'endstream' token after the specified length.
    """
    return _mupdf.ll_fz_open_endstream_filter(chain, len, offset)