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
def get_new_xref(self):
    """Make new xref."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    xref = 0
    ASSERT_PDF(pdf)
    ENSURE_OPERATION(pdf)
    xref = mupdf.pdf_create_object(pdf)
    return xref