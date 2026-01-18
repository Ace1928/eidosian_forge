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
def pdf_process_contents_outparams_fn(proc, doc, res, stm, cookie):
    """
    Class-aware helper for out-params of pdf_process_contents() [pdf_process_contents()].
    """
    out_res = ll_pdf_process_contents(proc.m_internal, doc.m_internal, res.m_internal, stm.m_internal, cookie.m_internal)
    return PdfObj(ll_pdf_keep_obj(out_res))