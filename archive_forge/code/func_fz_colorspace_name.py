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
def fz_colorspace_name(self):
    """
        Class-aware wrapper for `::fz_colorspace_name()`.
        	Query the name of a colorspace.

        	The returned string has the same lifespan as the colorspace
        	does. Caller should not free it.
        """
    return _mupdf.FzColorspace_fz_colorspace_name(self)