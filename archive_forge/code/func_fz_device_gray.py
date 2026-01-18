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
def fz_device_gray():
    """
    Class-aware wrapper for `::fz_device_gray()`.
    	Retrieve global default colorspaces.

    	These return borrowed references that should not be dropped,
    	unless they are kept first.
    """
    return _mupdf.fz_device_gray()