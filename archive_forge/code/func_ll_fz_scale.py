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
def ll_fz_scale(sx, sy):
    """
    Low-level wrapper for `::fz_scale()`.
    Create a scaling matrix.

    The returned matrix is of the form [ sx 0 0 sy 0 0 ].

    m: Pointer to the matrix to populate

    sx, sy: Scaling factors along the X- and Y-axes. A scaling
    factor of 1.0 will not cause any scaling along the relevant
    axis.

    Returns m.
    """
    return _mupdf.ll_fz_scale(sx, sy)