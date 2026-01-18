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
def pdf_edit_text_field_value(self, value, change, selStart, selEnd, newvalue):
    """
        Class-aware wrapper for `::pdf_edit_text_field_value()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_edit_text_field_value(const char *value, const char *change)` => `(int, int selStart, int selEnd, char *newvalue)`
        """
    return _mupdf.PdfAnnot_pdf_edit_text_field_value(self, value, change, selStart, selEnd, newvalue)