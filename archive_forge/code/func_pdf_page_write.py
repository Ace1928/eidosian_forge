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
def pdf_page_write(self, mediabox, presources, pcontents):
    """
        Class-aware wrapper for `::pdf_page_write()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_page_write(::fz_rect mediabox, ::pdf_obj **presources, ::fz_buffer **pcontents)` => `(fz_device *)`
        """
    return _mupdf.PdfDocument_pdf_page_write(self, mediabox, presources, pcontents)