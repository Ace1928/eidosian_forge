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
def pdf_process_contents(self, doc, res, stm, cookie, out_res):
    """
        Class-aware wrapper for `::pdf_process_contents()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_process_contents(::pdf_document *doc, ::pdf_obj *res, ::pdf_obj *stm, ::fz_cookie *cookie, ::pdf_obj **out_res)` =>
        """
    return _mupdf.PdfProcessor_pdf_process_contents(self, doc, res, stm, cookie, out_res)