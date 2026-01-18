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
def pdf_print_encrypted_obj(self, obj, tight, ascii, crypt, num, gen, sep):
    """
        Class-aware wrapper for `::pdf_print_encrypted_obj()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_print_encrypted_obj(::pdf_obj *obj, int tight, int ascii, ::pdf_crypt *crypt, int num, int gen)` => int sep
        """
    return _mupdf.FzOutput_pdf_print_encrypted_obj(self, obj, tight, ascii, crypt, num, gen, sep)