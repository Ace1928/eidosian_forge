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
def fz_close_document_writer(self):
    """
        Class-aware wrapper for `::fz_close_document_writer()`.
        	Called to end the process of writing
        	pages to a document.

        	This writes any file level trailers required. After this
        	completes successfully the file is up to date and complete.
        """
    return _mupdf.FzDocumentWriter_fz_close_document_writer(self)