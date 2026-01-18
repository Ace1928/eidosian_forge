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
def is_pdf(self):
    """Check for PDF."""
    if isinstance(self.this, mupdf.PdfDocument):
        return True
    if mupdf.ll_pdf_specifics(self.this.m_internal):
        ret = True
    else:
        ret = False
    return ret