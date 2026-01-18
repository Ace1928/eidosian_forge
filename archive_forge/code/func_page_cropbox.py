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
def page_cropbox(self, pno):
    """Get CropBox of page number (without loading page)."""
    if self.is_closed:
        raise ValueError('document closed')
    this_doc = self.this
    page_count = mupdf.fz_count_pages(this_doc)
    n = pno
    while n < 0:
        n += page_count
    pdf = _as_pdf_document(self)
    if n >= page_count:
        raise ValueError(MSG_BAD_PAGENO)
    ASSERT_PDF(pdf)
    pageref = mupdf.pdf_lookup_page_obj(pdf, n)
    cropbox = JM_cropbox(pageref)
    val = JM_py_from_rect(cropbox)
    val = Rect(val)
    return val