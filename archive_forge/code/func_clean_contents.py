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
def clean_contents(self, sanitize=1):
    if not sanitize and (not self.is_wrapped):
        self.wrap_contents()
    page = mupdf.pdf_page_from_fz_page(self.this)
    if not page.m_internal:
        return
    filter_ = _make_PdfFilterOptions(recurse=1, sanitize=sanitize)
    mupdf.pdf_filter_page_contents(page.doc(), page, filter_)