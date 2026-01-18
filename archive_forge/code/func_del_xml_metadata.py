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
def del_xml_metadata(self):
    """Delete XML metadata."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    if root.m_internal:
        mupdf.pdf_dict_del(root, PDF_NAME('Metadata'))