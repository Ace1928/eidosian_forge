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
def journal_save(self, filename):
    """Save journal to a file."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    if isinstance(filename, str):
        mupdf.pdf_save_journal(pdf, filename)
    else:
        out = JM_new_output_fileptr(filename)
        mupdf.pdf_write_journal(pdf, out)
        out.fz_close_output()