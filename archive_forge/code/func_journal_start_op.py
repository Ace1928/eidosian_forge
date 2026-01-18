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
def journal_start_op(self, name=None):
    """Begin a journalling operation."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    if not pdf.m_internal.journal:
        raise RuntimeError('Journalling not enabled')
    if name:
        mupdf.pdf_begin_operation(pdf, name)
    else:
        mupdf.pdf_begin_implicit_operation(pdf)