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
def pdfocr_tobytes(self, compress=True, language='eng', tessdata=None):
    """Save pixmap as an OCR-ed PDF page.

        Args:
            compress: (bool) compress, default 1 (True).
            language: (str) language(s) occurring on page, default "eng" (English),
                    multiples like "eng+ger" for English and German.
            tessdata: (str) folder name of Tesseract's language support. Must be
                    given if environment variable TESSDATA_PREFIX is not set.
        Notes:
            On failure, make sure Tesseract is installed and you have set the
            environment variable "TESSDATA_PREFIX" to the folder containing your
            Tesseract's language support data.
        """
    if not TESSDATA_PREFIX and (not tessdata):
        raise RuntimeError('No OCR support: TESSDATA_PREFIX not set')
    from io import BytesIO
    bio = BytesIO()
    self.pdfocr_save(bio, compress=compress, language=language, tessdata=tessdata)
    return bio.getvalue()