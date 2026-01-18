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
def fz_new_icc_colorspace(type, flags, name, buf):
    """
    Class-aware wrapper for `::fz_new_icc_colorspace()`.
    	Create a colorspace from an ICC profile supplied in buf.

    	Limited checking is done to ensure that the colorspace type is
    	appropriate for the supplied ICC profile.

    	An additional reference is taken to buf, which will be dropped
    	on destruction. Ownership is NOT passed in.

    	The returned reference should be dropped when it is finished
    	with.

    	Colorspaces are immutable once created.
    """
    return _mupdf.fz_new_icc_colorspace(type, flags, name, buf)