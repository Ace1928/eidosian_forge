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
def ll_fz_resolve_link_dest(doc, uri):
    """
    Low-level wrapper for `::fz_resolve_link_dest()`.
    Resolve an internal link to a page number, location, and possible viewing parameters.

    Returns location (-1,-1) if the URI cannot be resolved.
    """
    return _mupdf.ll_fz_resolve_link_dest(doc, uri)