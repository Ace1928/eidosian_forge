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
def fz_font_ft_face(self):
    """
        Class-aware wrapper for `::fz_font_ft_face()`.
        	Retrieve the FT_Face handle
        	for the font.

        	font: The font to query

        	Returns the FT_Face handle for the font, or NULL
        	if not a freetype handled font. (Cast to void *
        	to avoid nasty header exposure).
        """
    return _mupdf.FzFont_fz_font_ft_face(self)