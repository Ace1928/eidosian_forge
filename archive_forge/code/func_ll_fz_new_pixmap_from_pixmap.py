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
def ll_fz_new_pixmap_from_pixmap(pixmap, rect):
    """
    Low-level wrapper for `::fz_new_pixmap_from_pixmap()`.
    Create a new pixmap that represents a subarea of the specified
    pixmap. A reference is taken to this pixmap that will be dropped
    on destruction.

    The supplied rectangle must be wholly contained within the
    original pixmap.

    Returns a pointer to the new pixmap. Throws exception on failure
    to allocate.
    """
    return _mupdf.ll_fz_new_pixmap_from_pixmap(pixmap, rect)