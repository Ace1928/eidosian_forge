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
def pdf_field_event_validate(self, field, value, newvalue):
    """
        Class-aware wrapper for `::pdf_field_event_validate()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_field_event_validate(::pdf_obj *field, const char *value)` => `(int, char *newvalue)`
        """
    return _mupdf.PdfDocument_pdf_field_event_validate(self, field, value, newvalue)