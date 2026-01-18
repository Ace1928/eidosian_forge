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
def ll_pdf_signature_contents(doc, signature):
    """
    Wrapper for out-params of pdf_signature_contents().
    Returns: size_t, char *contents
    """
    outparams = ll_pdf_signature_contents_outparams()
    ret = ll_pdf_signature_contents_outparams_fn(doc, signature, outparams)
    return (ret, outparams.contents)