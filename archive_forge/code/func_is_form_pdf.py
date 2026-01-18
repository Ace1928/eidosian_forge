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
@property
def is_form_pdf(self):
    """Either False or PDF field count."""
    pdf = _as_pdf_document(self)
    if not pdf:
        return False
    count = -1
    try:
        fields = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), mupdf.PDF_ENUM_NAME_Root, mupdf.PDF_ENUM_NAME_AcroForm, mupdf.PDF_ENUM_NAME_Fields)
        if mupdf.pdf_is_array(fields):
            count = mupdf.pdf_array_len(fields)
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        return False
    if count >= 0:
        return count
    return False