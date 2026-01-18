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
def ll_fz_bound_glyph(font, gid, trm):
    """
    Low-level wrapper for `::fz_bound_glyph()`.
    Return a bbox for a given glyph in a font.

    font: The font to look for the glyph in.

    gid: The glyph to bound.

    trm: The matrix to apply to the glyph before bounding.

    Returns rectangle by value containing the bounds of the given
    glyph.
    """
    return _mupdf.ll_fz_bound_glyph(font, gid, trm)