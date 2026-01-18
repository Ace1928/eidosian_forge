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
def ll_fz_base_colorspace(cs):
    """
    Low-level wrapper for `::fz_base_colorspace()`.
    Get the 'base' colorspace for a colorspace.

    For indexed colorspaces, this is the colorspace the index
    decodes into. For all other colorspaces, it is the colorspace
    itself.

    The returned colorspace is 'borrowed' (i.e. no additional
    references are taken or dropped).
    """
    return _mupdf.ll_fz_base_colorspace(cs)