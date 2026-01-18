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
def fz_open_jbig2d(self, globals, embedded):
    """
        Class-aware wrapper for `::fz_open_jbig2d()`.
        	Open a filter that performs jbig2 decompression on the chained
        	stream, using the optional globals record.
        """
    return _mupdf.FzStream_fz_open_jbig2d(self, globals, embedded)