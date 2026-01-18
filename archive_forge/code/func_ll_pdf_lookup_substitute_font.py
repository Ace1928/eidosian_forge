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
def ll_pdf_lookup_substitute_font(mono, serif, bold, italic):
    """
    Wrapper for out-params of pdf_lookup_substitute_font().
    Returns: const unsigned char *, int len
    """
    outparams = ll_pdf_lookup_substitute_font_outparams()
    ret = ll_pdf_lookup_substitute_font_outparams_fn(mono, serif, bold, italic, outparams)
    return (ret, outparams.len)