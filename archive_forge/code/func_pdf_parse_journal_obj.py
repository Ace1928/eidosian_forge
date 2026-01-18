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
def pdf_parse_journal_obj(self, stm, onum, ostm, newobj):
    """
        Class-aware wrapper for `::pdf_parse_journal_obj()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_parse_journal_obj(::fz_stream *stm, ::fz_buffer **ostm)` => `(pdf_obj *, int onum, int newobj)`
        """
    return _mupdf.PdfDocument_pdf_parse_journal_obj(self, stm, onum, ostm, newobj)