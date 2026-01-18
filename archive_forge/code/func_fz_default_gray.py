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
def fz_default_gray(self):
    """
        Class-aware wrapper for `::fz_default_gray()`.
        	Retrieve default colorspaces (typically page local).

        	If default_cs is non NULL, the default is retrieved from there,
        	otherwise the global default is retrieved.

        	These return borrowed references that should not be dropped,
        	unless they are kept first.
        """
    return _mupdf.FzDefaultColorspaces_fz_default_gray(self)