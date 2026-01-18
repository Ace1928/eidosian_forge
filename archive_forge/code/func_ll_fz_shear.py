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
def ll_fz_shear(sx, sy):
    """
    Low-level wrapper for `::fz_shear()`.
    Create a shearing matrix.

    The returned matrix is of the form [ 1 sy sx 1 0 0 ].

    m: pointer to place to store returned matrix

    sx, sy: Shearing factors. A shearing factor of 0.0 will not
    cause any shearing along the relevant axis.

    Returns m.
    """
    return _mupdf.ll_fz_shear(sx, sy)