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
def ll_fz_new_pixmap_with_data(colorspace, w, h, seps, alpha, stride, samples):
    """
    Low-level wrapper for `::fz_new_pixmap_with_data()`.
    Create a new pixmap, with its origin at
    (0,0) using the supplied data block.

    cs: The colorspace to use for the pixmap, or NULL for an alpha
    plane/mask.

    w: The width of the pixmap (in pixels)

    h: The height of the pixmap (in pixels)

    seps: Details of separations.

    alpha: 0 for no alpha, 1 for alpha.

    stride: The byte offset from the pixel data in a row to the
    pixel data in the next row.

    samples: The data block to keep the samples in.

    Returns a pointer to the new pixmap. Throws exception on failure to
    allocate.
    """
    return _mupdf.ll_fz_new_pixmap_with_data(colorspace, w, h, seps, alpha, stride, samples)