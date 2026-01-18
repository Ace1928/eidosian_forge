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
def ll_fz_new_xhtml_document_from_document(old_doc, opts):
    """
    Low-level wrapper for `::fz_new_xhtml_document_from_document()`.
    Use text extraction to convert the input document into XHTML,
    then open the result as a new document that can be reflowed.
    """
    return _mupdf.ll_fz_new_xhtml_document_from_document(old_doc, opts)