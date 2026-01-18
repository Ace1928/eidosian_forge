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
def pdf_parse_ind_obj(self, f, num, gen, stm_ofs, try_repair):
    """
        Class-aware wrapper for `::pdf_parse_ind_obj()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_parse_ind_obj(::fz_stream *f)` => `(pdf_obj *, int num, int gen, int64_t stm_ofs, int try_repair)`
        """
    return _mupdf.PdfDocument_pdf_parse_ind_obj(self, f, num, gen, stm_ofs, try_repair)