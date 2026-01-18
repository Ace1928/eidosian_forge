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
def ll_pdf_signature_info(name, dn, reason, location, date, include_labels):
    """ Low-level wrapper for `::pdf_signature_info()`."""
    return _mupdf.ll_pdf_signature_info(name, dn, reason, location, date, include_labels)