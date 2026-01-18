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
def get_sigflags(self):
    """Get the /SigFlags value."""
    pdf = _as_pdf_document(self)
    if not pdf:
        return -1
    sigflags = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('AcroForm'), PDF_NAME('SigFlags'))
    sigflag = -1
    if sigflags.m_internal:
        sigflag = mupdf.pdf_to_int(sigflags)
    return sigflag