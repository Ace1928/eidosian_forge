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
def ll_fz_register_document_handler(handler):
    """
    Low-level wrapper for `::fz_register_document_handler()`.
    Register a handler for a document type.

    handler: The handler to register.
    """
    return _mupdf.ll_fz_register_document_handler(handler)