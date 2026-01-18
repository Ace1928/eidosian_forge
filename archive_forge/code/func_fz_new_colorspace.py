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
def fz_new_colorspace(type, flags, n, name):
    """
    Class-aware wrapper for `::fz_new_colorspace()`.
    	Creates a new colorspace instance and returns a reference.

    	No internal checking is done that the colorspace type (e.g.
    	CMYK) matches with the flags (e.g. FZ_COLORSPACE_HAS_CMYK) or
    	colorant count (n) or name.

    	The reference should be dropped when it is finished with.

    	Colorspaces are immutable once created (with the exception of
    	setting up colorant names for separation spaces).
    """
    return _mupdf.fz_new_colorspace(type, flags, n, name)