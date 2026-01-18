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
def pdf_js_event_result_validate(self, newvalue):
    """
        Class-aware wrapper for `::pdf_js_event_result_validate()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_js_event_result_validate()` => `(int, char *newvalue)`
        """
    return _mupdf.PdfJs_pdf_js_event_result_validate(self, newvalue)