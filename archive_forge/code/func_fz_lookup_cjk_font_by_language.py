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
def fz_lookup_cjk_font_by_language(lang, len, subfont):
    """
    Class-aware wrapper for `::fz_lookup_cjk_font_by_language()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_lookup_cjk_font_by_language(const char *lang)` => `(const unsigned char *, int len, int subfont)`

    	Search the builtin cjk fonts for a match for a given language.
    	Whether a font is present or not will depend on the
    	configuration in which MuPDF is built.

    	lang: Pointer to a (case sensitive) language string (e.g.
    	"ja", "ko", "zh-Hant" etc).

    	len: Pointer to a place to receive the length of the discovered
    	font buffer.

    	subfont: Pointer to a place to store the subfont index of the
    	discovered font.

    	Returns a pointer to the font file data, or NULL if not present.
    """
    return _mupdf.fz_lookup_cjk_font_by_language(lang, len, subfont)