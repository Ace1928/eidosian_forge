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
def fz_currentpoint(self):
    """
        Class-aware wrapper for `::fz_currentpoint()`.
        	Return the current point that a path has
        	reached or (0,0) if empty.

        	path: path to return the current point of.
        """
    return _mupdf.FzPath_fz_currentpoint(self)