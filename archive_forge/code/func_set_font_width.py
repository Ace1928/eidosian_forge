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
@staticmethod
def set_font_width(doc, xref, width):
    pdf = _as_pdf_document(doc)
    if not pdf:
        return False
    font = mupdf.pdf_load_object(pdf, xref)
    dfonts = mupdf.pdf_dict_get(font, PDF_NAME('DescendantFonts'))
    if mupdf.pdf_is_array(dfonts):
        n = mupdf.pdf_array_len(dfonts)
        for i in range(n):
            dfont = mupdf.pdf_array_get(dfonts, i)
            warray = mupdf.pdf_new_array(pdf, 3)
            mupdf.pdf_array_push(warray, mupdf.pdf_new_int(0))
            mupdf.pdf_array_push(warray, mupdf.pdf_new_int(65535))
            mupdf.pdf_array_push(warray, mupdf.pdf_new_int(width))
            mupdf.pdf_dict_put(dfont, PDF_NAME('W'), warray)
    return True