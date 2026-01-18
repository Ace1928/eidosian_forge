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
def get_ocgs(self):
    """Show existing optional content groups."""
    ci = mupdf.pdf_new_name('CreatorInfo')
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    ocgs = mupdf.pdf_dict_getl(mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root')), PDF_NAME('OCProperties'), PDF_NAME('OCGs'))
    rc = dict()
    if not mupdf.pdf_is_array(ocgs):
        return rc
    n = mupdf.pdf_array_len(ocgs)
    for i in range(n):
        ocg = mupdf.pdf_array_get(ocgs, i)
        xref = mupdf.pdf_to_num(ocg)
        name = mupdf.pdf_to_text_string(mupdf.pdf_dict_get(ocg, PDF_NAME('Name')))
        obj = mupdf.pdf_dict_getl(ocg, PDF_NAME('Usage'), ci, PDF_NAME('Subtype'))
        usage = None
        if obj.m_internal:
            usage = mupdf.pdf_to_name(obj)
        intents = list()
        intent = mupdf.pdf_dict_get(ocg, PDF_NAME('Intent'))
        if intent.m_internal:
            if mupdf.pdf_is_name(intent):
                intents.append(mupdf.pdf_to_name(intent))
            elif mupdf.pdf_is_array(intent):
                m = mupdf.pdf_array_len(intent)
                for j in range(m):
                    o = mupdf.pdf_array_get(intent, j)
                    if mupdf.pdf_is_name(o):
                        intents.append(mupdf.pdf_to_name(o))
        hidden = mupdf.pdf_is_ocg_hidden(pdf, mupdf.PdfObj(), usage, ocg)
        item = {'name': name, 'intent': intents, 'on': not hidden, 'usage': usage}
        temp = xref
        rc[temp] = item
    return rc