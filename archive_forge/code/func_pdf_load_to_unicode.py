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
def pdf_load_to_unicode(self, font, strings, collection, cmapstm):
    """
        Class-aware wrapper for `::pdf_load_to_unicode()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_load_to_unicode(::pdf_font_desc *font, char *collection, ::pdf_obj *cmapstm)` => const char *strings
        """
    return _mupdf.PdfDocument_pdf_load_to_unicode(self, font, strings, collection, cmapstm)