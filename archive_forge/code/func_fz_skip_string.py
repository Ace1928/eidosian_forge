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
def fz_skip_string(self, str):
    """
        Class-aware wrapper for `::fz_skip_string()`.
        	Skip over a given string in a stream. Return 0 if successfully
        	skipped, non-zero otherwise. As many characters will be skipped
        	over as matched in the string.
        """
    return _mupdf.FzStream_fz_skip_string(self, str)