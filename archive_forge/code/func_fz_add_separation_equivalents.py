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
def fz_add_separation_equivalents(self, rgba, cmyk, name):
    """
        Class-aware wrapper for `::fz_add_separation_equivalents()`.
        	Add a separation with equivalents (null terminated name,
        	colorspace)

        	(old, deprecated)
        """
    return _mupdf.FzSeparations_fz_add_separation_equivalents(self, rgba, cmyk, name)