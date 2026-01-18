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
def pdf_obj_memo(self, bit, memo):
    """
        Class-aware wrapper for `::pdf_obj_memo()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_obj_memo(int bit)` => `(int, int memo)`
        """
    return _mupdf.PdfObj_pdf_obj_memo(self, bit, memo)