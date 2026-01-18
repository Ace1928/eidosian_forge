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
def page_xref(self, pno):
    """Get xref of page number."""
    if g_use_extra:
        return extra.page_xref(self.this, pno)
    if self.is_closed:
        raise ValueError('document closed')
    page_count = mupdf.fz_count_pages(self.this)
    n = pno
    while n < 0:
        n += page_count
    pdf = _as_pdf_document(self)
    xref = 0
    if n >= page_count:
        raise ValueError(MSG_BAD_PAGENO)
    ASSERT_PDF(pdf)
    xref = mupdf.pdf_to_num(mupdf.pdf_lookup_page_obj(pdf, n))
    return xref