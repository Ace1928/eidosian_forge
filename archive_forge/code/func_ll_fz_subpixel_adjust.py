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
def ll_fz_subpixel_adjust(ctm, subpix_ctm, qe, qf):
    """
    Low-level wrapper for `::fz_subpixel_adjust()`.
    Perform subpixel quantisation and adjustment on a glyph matrix.

    ctm: On entry, the desired 'ideal' transformation for a glyph.
    On exit, adjusted to a (very similar) transformation quantised
    for subpixel caching.

    subpix_ctm: Initialised by the routine to the transform that
    should be used to render the glyph.

    qe, qf: which subpixel position we quantised to.

    Returns: the size of the glyph.

    Note: This is currently only exposed for use in our app. It
    should be considered "at risk" of removal from the API.
    """
    return _mupdf.ll_fz_subpixel_adjust(ctm, subpix_ctm, qe, qf)