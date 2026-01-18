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
def fz_closepath(self):
    """
        Class-aware wrapper for `::fz_closepath()`.
        	Close the current subpath.

        	path: The path to modify.

        	Throws exceptions on failure to allocate, attempting to modify
        	a packed path, and illegal path closes (i.e. closing a non open
        	path).
        """
    return _mupdf.FzPath_fz_closepath(self)