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
def pdfocr_save(self, filename, compress=1, language=None, tessdata=None):
    """
        Save pixmap as an OCR-ed PDF page.
        """
    if not TESSDATA_PREFIX and (not tessdata):
        raise RuntimeError('No OCR support: TESSDATA_PREFIX not set')
    opts = mupdf.FzPdfocrOptions()
    opts.compress = compress
    if language:
        opts.language_set2(language)
    if tessdata:
        opts.datadir_set2(tessdata)
    pix = self.this
    if isinstance(filename, str):
        mupdf.fz_save_pixmap_as_pdfocr(pix, filename, 0, opts)
    else:
        out = JM_new_output_fileptr(filename)
        mupdf.fz_write_pixmap_as_pdfocr(out, pix, opts)
        out.fz_close_output()