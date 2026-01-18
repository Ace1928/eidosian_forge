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
def pdf_dict_get_string(self, key, sizep):
    """
        Class-aware wrapper for `::pdf_dict_get_string()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_dict_get_string(::pdf_obj *key)` => `(const char *, size_t sizep)`
        """
    return _mupdf.PdfObj_pdf_dict_get_string(self, key, sizep)