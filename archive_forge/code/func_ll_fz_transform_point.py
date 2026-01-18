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
def ll_fz_transform_point(point, m):
    """
    Low-level wrapper for `::fz_transform_point()`.
    Apply a transformation to a point.

    transform: Transformation matrix to apply. See fz_concat,
    fz_scale, fz_rotate and fz_translate for how to create a
    matrix.

    point: Pointer to point to update.

    Returns transform (unchanged).
    """
    return _mupdf.ll_fz_transform_point(point, m)