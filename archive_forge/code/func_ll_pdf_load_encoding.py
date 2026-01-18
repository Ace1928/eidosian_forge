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
def ll_pdf_load_encoding(encoding):
    """
    Wrapper for out-params of pdf_load_encoding().
    Returns: const char *estrings
    """
    outparams = ll_pdf_load_encoding_outparams()
    ret = ll_pdf_load_encoding_outparams_fn(encoding, outparams)
    return outparams.estrings