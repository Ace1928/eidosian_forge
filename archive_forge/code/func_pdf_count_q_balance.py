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
def pdf_count_q_balance(self, res, stm, underflow, overflow):
    """
        Class-aware wrapper for `::pdf_count_q_balance()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_count_q_balance(::pdf_obj *res, ::pdf_obj *stm)` => `(int underflow, int overflow)`
        """
    return _mupdf.PdfDocument_pdf_count_q_balance(self, res, stm, underflow, overflow)