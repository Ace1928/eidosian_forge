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
def pdf_annot_MK_BG(self, n, color):
    """
        Class-aware wrapper for `::pdf_annot_MK_BG()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_annot_MK_BG(float color[4])` => int n
        """
    return _mupdf.PdfAnnot_pdf_annot_MK_BG(self, n, color)