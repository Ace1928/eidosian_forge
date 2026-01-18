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
def ll_pdf_load_object(doc, num):
    """
    Low-level wrapper for `::pdf_load_object()`.
    Load a given object.

    This can cause xref reorganisations (solidifications etc) due to
    repairs, so all held pdf_xref_entries should be considered
    invalid after this call (other than the returned one).
    """
    return _mupdf.ll_pdf_load_object(doc, num)