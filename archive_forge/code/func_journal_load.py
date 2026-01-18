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
def journal_load(self, filename):
    """Load a journal from a file."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    if isinstance(filename, str):
        mupdf.pdf_load_journal(pdf, filename)
    else:
        res = JM_BufferFromBytes(filename)
        stm = mupdf.fz_open_buffer(res)
        mupdf.pdf_deserialise_journal(pdf, stm)
    if not pdf.m_internal.journal:
        RAISEPY('Journal and document do not match', JM_Exc_FileDataError)