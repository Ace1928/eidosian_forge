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
def extract_font(self, xref=0, info_only=0, named=None):
    """
        Get a font by xref. Returns a tuple or dictionary.
        """
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    obj = mupdf.pdf_load_object(pdf, xref)
    type_ = mupdf.pdf_dict_get(obj, PDF_NAME('Type'))
    subtype = mupdf.pdf_dict_get(obj, PDF_NAME('Subtype'))
    if mupdf.pdf_name_eq(type_, PDF_NAME('Font')) and (not mupdf.pdf_to_name(subtype).startswith('CIDFontType')):
        basefont = mupdf.pdf_dict_get(obj, PDF_NAME('BaseFont'))
        if not basefont.m_internal or mupdf.pdf_is_null(basefont):
            bname = mupdf.pdf_dict_get(obj, PDF_NAME('Name'))
        else:
            bname = basefont
        ext = JM_get_fontextension(pdf, xref)
        if ext != 'n/a' and (not info_only):
            buffer_ = JM_get_fontbuffer(pdf, xref)
            bytes_ = JM_BinFromBuffer(buffer_)
        else:
            bytes_ = b''
        if not named:
            rc = (JM_EscapeStrFromStr(mupdf.pdf_to_name(bname)), JM_UnicodeFromStr(ext), JM_UnicodeFromStr(mupdf.pdf_to_name(subtype)), bytes_)
        else:
            rc = {dictkey_name: JM_EscapeStrFromStr(mupdf.pdf_to_name(bname)), dictkey_ext: JM_UnicodeFromStr(ext), dictkey_type: JM_UnicodeFromStr(mupdf.pdf_to_name(subtype)), dictkey_content: bytes_}
    elif not named:
        rc = ('', '', '', b'')
    else:
        rc = {dictkey_name: '', dictkey_ext: '', dictkey_type: '', dictkey_content: b''}
    return rc