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
def ll_fz_open_range_filter(chain, ranges, nranges):
    """
    Low-level wrapper for `::fz_open_range_filter()`.
    The range filter copies data from specified ranges of the
    chained stream.
    """
    return _mupdf.ll_fz_open_range_filter(chain, ranges, nranges)