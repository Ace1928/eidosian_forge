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
def fz_arc4_init(self, key, len):
    """
        Class-aware wrapper for `::fz_arc4_init()`.
        	RC4 initialization. Begins an RC4 operation, writing a new
        	context.

        	Never throws an exception.
        """
    return _mupdf.FzArc4_fz_arc4_init(self, key, len)