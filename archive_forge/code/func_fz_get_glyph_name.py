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
def fz_get_glyph_name(self, glyph, buf, size):
    """
        Class-aware wrapper for `::fz_get_glyph_name()`.
        	Find the name of a glyph

        	font: The font to look for the glyph in.

        	glyph: The glyph id to look for.

        	buf: Pointer to a buffer for the name to be inserted into.

        	size: The size of the buffer.

        	If a font contains a name table, then the name of the glyph
        	will be returned in the supplied buffer. Otherwise a name
        	is synthesised. The name will be truncated to fit in
        	the buffer.
        """
    return _mupdf.FzFont_fz_get_glyph_name(self, glyph, buf, size)