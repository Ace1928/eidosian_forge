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
def ll_fz_transform_rect(rect, m):
    """
    Low-level wrapper for `::fz_transform_rect()`.
    Apply a transform to a rectangle.

    After the four corner points of the axis-aligned rectangle
    have been transformed it may not longer be axis-aligned. So a
    new axis-aligned rectangle is created covering at least the
    area of the transformed rectangle.

    transform: Transformation matrix to apply. See fz_concat,
    fz_scale and fz_rotate for how to create a matrix.

    rect: Rectangle to be transformed. The two special cases
    fz_empty_rect and fz_infinite_rect, may be used but are
    returned unchanged as expected.
    """
    return _mupdf.ll_fz_transform_rect(rect, m)