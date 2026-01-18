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
def fz_render_t3_glyph_direct(self, font, gid, trm, gstate, def_cs):
    """
        Class-aware wrapper for `::fz_render_t3_glyph_direct()`.
        	Nasty PDF interpreter specific hernia, required to allow the
        	interpreter to replay glyphs from a type3 font directly into
        	the target device.

        	This is only used in exceptional circumstances (such as type3
        	glyphs that inherit current graphics state, or nested type3
        	glyphs).
        """
    return _mupdf.FzDevice_fz_render_t3_glyph_direct(self, font, gid, trm, gstate, def_cs)