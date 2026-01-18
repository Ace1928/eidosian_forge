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
def mediabox(self):
    """The MediaBox."""
    CheckParent(self)
    page = self._pdf_page()
    if not page.m_internal:
        rect = mupdf.fz_bound_page(self.this)
    else:
        rect = JM_mediabox(page.obj())
    return Rect(rect)