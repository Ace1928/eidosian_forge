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
def ll_fz_encode_character_by_glyph_name(font, glyphname):
    """
    Low-level wrapper for `::fz_encode_character_by_glyph_name()`.
    Encode character.

    Either by direct lookup of glyphname within a font, or, failing
    that, by mapping glyphname to unicode and thence to the glyph
    index within the given font.

    Returns zero for type3 fonts.
    """
    return _mupdf.ll_fz_encode_character_by_glyph_name(font, glyphname)