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
def ll_fz_prepare_t3_glyph(font, gid):
    """
    Low-level wrapper for `::fz_prepare_t3_glyph()`.
    Force a type3 font to cache the displaylist for a given glyph
    id.

    This caching can involve reading the underlying file, so must
    happen ahead of time, so we aren't suddenly forced to read the
    file while playing a displaylist back.
    """
    return _mupdf.ll_fz_prepare_t3_glyph(font, gid)