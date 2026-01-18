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
def journal_can_do(self):
    """Show if undo and / or redo are possible."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    undo = 0
    redo = 0
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    undo = mupdf.pdf_can_undo(pdf)
    redo = mupdf.pdf_can_redo(pdf)
    return {'undo': bool(undo), 'redo': bool(redo)}