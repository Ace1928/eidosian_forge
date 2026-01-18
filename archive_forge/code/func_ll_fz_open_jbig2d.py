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
def ll_fz_open_jbig2d(chain, globals, embedded):
    """
    Low-level wrapper for `::fz_open_jbig2d()`.
    Open a filter that performs jbig2 decompression on the chained
    stream, using the optional globals record.
    """
    return _mupdf.ll_fz_open_jbig2d(chain, globals, embedded)