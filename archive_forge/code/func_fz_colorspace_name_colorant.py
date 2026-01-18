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
def fz_colorspace_name_colorant(self, n, name):
    """
        Class-aware wrapper for `::fz_colorspace_name_colorant()`.
        	Assign a name for a given colorant in a colorspace.

        	Used while initially setting up a colorspace. The string is
        	copied into local storage, so need not be retained by the
        	caller.
        """
    return _mupdf.FzColorspace_fz_colorspace_name_colorant(self, n, name)