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
def page_annot_xrefs(self, n):
    if g_use_extra:
        return extra.page_annot_xrefs(self.this, n)
    if isinstance(self.this, mupdf.PdfDocument):
        page_count = mupdf.pdf_count_pages(self.this)
        pdf_document = self.this
    else:
        page_count = mupdf.fz_count_pages(self.this)
        pdf_document = _as_pdf_document(self)
    while n < 0:
        n += page_count
    if n > page_count:
        raise ValueError(MSG_BAD_PAGENO)
    page_obj = mupdf.pdf_lookup_page_obj(pdf_document, n)
    annots = JM_get_annot_xref_list(page_obj)
    return annots