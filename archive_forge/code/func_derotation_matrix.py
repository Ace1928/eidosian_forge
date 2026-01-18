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
def derotation_matrix(self) -> Matrix:
    """Reflects page de-rotation."""
    if g_use_extra:
        return Matrix(extra.Page_derotate_matrix(self.this))
    pdfpage = self._pdf_page()
    if not pdfpage.m_internal:
        return Matrix(mupdf.FzRect(mupdf.FzRect.UNIT))
    return Matrix(JM_derotate_page_matrix(pdfpage))