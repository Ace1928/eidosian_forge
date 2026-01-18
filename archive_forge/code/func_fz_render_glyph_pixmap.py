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
def fz_render_glyph_pixmap(self, gid, ctm, scissor, aa):
    """
        Class-aware wrapper for `::fz_render_glyph_pixmap()`.
        	Create a pixmap containing a rendered glyph.

        	Lookup gid from font, clip it with scissor, and rendering it
        	with aa bits of antialiasing into a new pixmap.

        	The caller takes ownership of the pixmap and so must free it.

        	Note: This function is no longer used for normal rendering
        	operations, and is kept around just because we use it in the
        	app. It should be considered "at risk" of removal from the API.
        """
    return _mupdf.FzFont_fz_render_glyph_pixmap(self, gid, ctm, scissor, aa)