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
def ll_fz_bound_shade(shade, ctm):
    """
    Low-level wrapper for `::fz_bound_shade()`.
    Bound a given shading.

    shade: The shade to bound.

    ctm: The transform to apply to the shade before bounding.

    r: Pointer to storage to put the bounds in.

    Returns r, updated to contain the bounds for the shading.
    """
    return _mupdf.ll_fz_bound_shade(shade, ctm)