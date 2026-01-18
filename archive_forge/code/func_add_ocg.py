import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def add_ocg(self, name, config=-1, on=1, intent=None, usage=None):
    """Add new optional content group."""
    xref = 0
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    ocg = mupdf.pdf_add_new_dict(pdf, 3)
    mupdf.pdf_dict_put(ocg, PDF_NAME('Type'), PDF_NAME('OCG'))
    mupdf.pdf_dict_put_text_string(ocg, PDF_NAME('Name'), name)
    intents = mupdf.pdf_dict_put_array(ocg, PDF_NAME('Intent'), 2)
    if not intent:
        mupdf.pdf_array_push(intents, PDF_NAME('View'))
    elif not isinstance(intent, str):
        assert 0, f'fixme: intent is not a str. type(intent)={type(intent)!r} type={type!r}'
    else:
        mupdf.pdf_array_push(intents, mupdf.pdf_new_name(intent))
    use_for = mupdf.pdf_dict_put_dict(ocg, PDF_NAME('Usage'), 3)
    ci_name = mupdf.pdf_new_name('CreatorInfo')
    cre_info = mupdf.pdf_dict_put_dict(use_for, ci_name, 2)
    mupdf.pdf_dict_put_text_string(cre_info, PDF_NAME('Creator'), 'PyMuPDF')
    if usage:
        mupdf.pdf_dict_put_name(cre_info, PDF_NAME('Subtype'), usage)
    else:
        mupdf.pdf_dict_put_name(cre_info, PDF_NAME('Subtype'), 'Artwork')
    indocg = mupdf.pdf_add_object(pdf, ocg)
    ocp = JM_ensure_ocproperties(pdf)
    obj = mupdf.pdf_dict_get(ocp, PDF_NAME('OCGs'))
    mupdf.pdf_array_push(obj, indocg)
    if config > -1:
        obj = mupdf.pdf_dict_get(ocp, PDF_NAME('Configs'))
        if not mupdf.pdf_is_array(obj):
            raise ValueError(MSG_BAD_OC_CONFIG)
        cfg = mupdf.pdf_array_get(obj, config)
        if not cfg.m_internal:
            raise ValueError(MSG_BAD_OC_CONFIG)
    else:
        cfg = mupdf.pdf_dict_get(ocp, PDF_NAME('D'))
    obj = mupdf.pdf_dict_get(cfg, PDF_NAME('Order'))
    if not obj.m_internal:
        obj = mupdf.pdf_dict_put_array(cfg, PDF_NAME('Order'), 1)
    mupdf.pdf_array_push(obj, indocg)
    if on:
        obj = mupdf.pdf_dict_get(cfg, PDF_NAME('ON'))
        if not obj.m_internal:
            obj = mupdf.pdf_dict_put_array(cfg, PDF_NAME('ON'), 1)
    else:
        obj = mupdf.pdf_dict_get(cfg, PDF_NAME('OFF'))
        if not obj.m_internal:
            obj = mupdf.pdf_dict_put_array(cfg, PDF_NAME('OFF'), 1)
    mupdf.pdf_array_push(obj, indocg)
    mupdf.ll_pdf_read_ocg(pdf.m_internal)
    xref = mupdf.pdf_to_num(indocg)
    return xref