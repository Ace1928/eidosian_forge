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
def pdf_repair_obj(self, buf, stmofsp, stmlenp, encrypt, id, page, tmpofs, root):
    """
        Class-aware wrapper for `::pdf_repair_obj()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_repair_obj(::pdf_lexbuf *buf, ::pdf_obj **encrypt, ::pdf_obj **id, ::pdf_obj **page, ::pdf_obj **root)` => `(int, int64_t stmofsp, int64_t stmlenp, int64_t tmpofs)`
        """
    return _mupdf.PdfDocument_pdf_repair_obj(self, buf, stmofsp, stmlenp, encrypt, id, page, tmpofs, root)