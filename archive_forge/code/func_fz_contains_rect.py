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
def fz_contains_rect(self, b):
    """
        Class-aware wrapper for `::fz_contains_rect()`.
        	Test rectangle inclusion.

        	Return true if a entirely contains b.
        """
    return _mupdf.FzRect_fz_contains_rect(self, b)