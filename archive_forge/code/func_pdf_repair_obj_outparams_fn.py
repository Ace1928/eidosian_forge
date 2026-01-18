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
def pdf_repair_obj_outparams_fn(doc, buf):
    """
    Class-aware helper for out-params of pdf_repair_obj() [pdf_repair_obj()].
    """
    ret, stmofsp, stmlenp, encrypt, id, page, tmpofs, root = ll_pdf_repair_obj(doc.m_internal, buf.m_internal)
    return (ret, stmofsp, stmlenp, PdfObj(ll_pdf_keep_obj(encrypt)), PdfObj(ll_pdf_keep_obj(id)), PdfObj(ll_pdf_keep_obj(page)), tmpofs, PdfObj(ll_pdf_keep_obj(root)))