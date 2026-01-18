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
def ll_pdf_print_encrypted_obj(out, obj, tight, ascii, crypt, num, gen):
    """
    Wrapper for out-params of pdf_print_encrypted_obj().
    Returns: int sep
    """
    outparams = ll_pdf_print_encrypted_obj_outparams()
    ret = ll_pdf_print_encrypted_obj_outparams_fn(out, obj, tight, ascii, crypt, num, gen, outparams)
    return outparams.sep