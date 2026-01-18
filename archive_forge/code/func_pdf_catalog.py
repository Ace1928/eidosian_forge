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
def pdf_catalog(self):
    """Get xref of PDF catalog."""
    pdf = _as_pdf_document(self)
    xref = 0
    if not pdf:
        return xref
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    xref = mupdf.pdf_to_num(root)
    return xref