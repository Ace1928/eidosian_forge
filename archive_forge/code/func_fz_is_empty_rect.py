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
def fz_is_empty_rect(self):
    """
        Class-aware wrapper for `::fz_is_empty_rect()`.
        	Check if rectangle is empty.

        	An empty rectangle is defined as one whose area is zero.
        	All invalid rectangles are empty.
        """
    return _mupdf.FzRect_fz_is_empty_rect(self)