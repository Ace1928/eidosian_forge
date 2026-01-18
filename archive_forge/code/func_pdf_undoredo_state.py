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
def pdf_undoredo_state(self, steps):
    """
        Class-aware wrapper for `::pdf_undoredo_state()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_undoredo_state()` => `(int, int steps)`
        """
    return _mupdf.PdfDocument_pdf_undoredo_state(self, steps)