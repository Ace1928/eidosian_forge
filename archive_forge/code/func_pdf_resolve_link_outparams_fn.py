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
def pdf_resolve_link_outparams_fn(doc, uri):
    """
    Class-aware helper for out-params of pdf_resolve_link() [pdf_resolve_link()].
    """
    ret, xp, yp = ll_pdf_resolve_link(doc.m_internal, uri)
    return (ret, xp, yp)