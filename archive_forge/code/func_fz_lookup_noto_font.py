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
def fz_lookup_noto_font(script, lang, len, subfont):
    """
    Class-aware wrapper for `::fz_lookup_noto_font()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_lookup_noto_font(int script, int lang)` => `(const unsigned char *, int len, int subfont)`

    	Search the builtin noto fonts for a match.
    	Whether a font is present or not will depend on the
    	configuration in which MuPDF is built.

    	script: The script desired (e.g. UCDN_SCRIPT_KATAKANA).

    	lang: The language desired (e.g. FZ_LANG_ja).

    	len: Pointer to a place to receive the length of the discovered
    	font buffer.

    	Returns a pointer to the font file data, or NULL if not present.
    """
    return _mupdf.fz_lookup_noto_font(script, lang, len, subfont)