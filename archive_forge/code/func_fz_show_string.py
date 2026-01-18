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
def fz_show_string(self, font, trm, s, wmode, bidi_level, markup_dir, language):
    """
        Class-aware wrapper for `::fz_show_string()`.
        	Add a UTF8 string to a text object.

        	text: Text object to add to.

        	font: The font the string should be added in.

        	trm: The transform to use.

        	s: The utf-8 string to add.

        	wmode: 1 for vertical mode, 0 for horizontal.

        	bidi_level: The bidirectional level for this glyph.

        	markup_dir: The direction of the text as specified in the markup.

        	language: The language in use (if known, 0 otherwise)
        		(e.g. FZ_LANG_zh_Hans).

        	Returns the transform updated with the advance width of the
        	string.
        """
    return _mupdf.FzText_fz_show_string(self, font, trm, s, wmode, bidi_level, markup_dir, language)