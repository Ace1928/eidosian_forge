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
def pdf_annot_default_appearance_outparams_fn(annot, color):
    """
    Class-aware helper for out-params of pdf_annot_default_appearance() [pdf_annot_default_appearance()].
    """
    font, size, n = ll_pdf_annot_default_appearance(annot.m_internal, color)
    return (font, size, n)