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
def ll_pdf_page_write(doc, mediabox):
    """
    Wrapper for out-params of pdf_page_write().
    Returns: fz_device *, ::pdf_obj *presources, ::fz_buffer *pcontents
    """
    outparams = ll_pdf_page_write_outparams()
    ret = ll_pdf_page_write_outparams_fn(doc, mediabox, outparams)
    return (ret, outparams.presources, outparams.pcontents)