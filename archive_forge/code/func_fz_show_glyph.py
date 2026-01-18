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
def fz_show_glyph(self, font, trm, glyph, unicode, wmode, bidi_level, markup_dir, language):
    """
        Class-aware wrapper for `::fz_show_glyph()`.
        	Add a glyph/unicode value to a text object.

        	text: Text object to add to.

        	font: The font the glyph should be added in.

        	trm: The transform to use for the glyph.

        	glyph: The glyph id to add.

        	unicode: The unicode character for the glyph.

        	cid: The CJK CID value or raw character code.

        	wmode: 1 for vertical mode, 0 for horizontal.

        	bidi_level: The bidirectional level for this glyph.

        	markup_dir: The direction of the text as specified in the
        	markup.

        	language: The language in use (if known, 0 otherwise)
        	(e.g. FZ_LANG_zh_Hans).

        	Throws exception on failure to allocate.
        """
    return _mupdf.FzText_fz_show_glyph(self, font, trm, glyph, unicode, wmode, bidi_level, markup_dir, language)