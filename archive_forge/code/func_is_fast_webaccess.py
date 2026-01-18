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
def is_fast_webaccess(self):
    """
        Check whether we have a linearized PDF.
        """
    pdf = _as_pdf_document(self)
    if pdf.m_internal:
        return mupdf.pdf_doc_was_linearized(pdf)
    return False