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
def ll_fz_run_document_structure(doc, dev, cookie):
    """
    Low-level wrapper for `::fz_run_document_structure()`.
    Run the document structure through a device.

    doc: Document in question.

    dev: Device obtained from fz_new_*_device.

    cookie: Communication mechanism between caller and library.
    Intended for multi-threaded applications, while
    single-threaded applications set cookie to NULL. The
    caller may abort an ongoing rendering of a page. Cookie also
    communicates progress information back to the caller. The
    fields inside cookie are continually updated while the page is
    rendering.
    """
    return _mupdf.ll_fz_run_document_structure(doc, dev, cookie)