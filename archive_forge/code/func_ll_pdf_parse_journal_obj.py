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
def ll_pdf_parse_journal_obj(doc, stm):
    """
    Wrapper for out-params of pdf_parse_journal_obj().
    Returns: pdf_obj *, int onum, ::fz_buffer *ostm, int newobj
    """
    outparams = ll_pdf_parse_journal_obj_outparams()
    ret = ll_pdf_parse_journal_obj_outparams_fn(doc, stm, outparams)
    return (ret, outparams.onum, outparams.ostm, outparams.newobj)