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
def ll_pdf_load_to_unicode(doc, font, collection, cmapstm):
    """
    Wrapper for out-params of pdf_load_to_unicode().
    Returns: const char *strings
    """
    outparams = ll_pdf_load_to_unicode_outparams()
    ret = ll_pdf_load_to_unicode_outparams_fn(doc, font, collection, cmapstm, outparams)
    return outparams.strings