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
def ll_fz_document_supports_accelerator(doc):
    """
    Low-level wrapper for `::fz_document_supports_accelerator()`.
    Query if the document supports the saving of accelerator data.
    """
    return _mupdf.ll_fz_document_supports_accelerator(doc)