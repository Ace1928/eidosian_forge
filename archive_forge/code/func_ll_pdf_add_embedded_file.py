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
def ll_pdf_add_embedded_file(doc, filename, mimetype, contents, created, modifed, add_checksum):
    """ Low-level wrapper for `::pdf_add_embedded_file()`."""
    return _mupdf.ll_pdf_add_embedded_file(doc, filename, mimetype, contents, created, modifed, add_checksum)