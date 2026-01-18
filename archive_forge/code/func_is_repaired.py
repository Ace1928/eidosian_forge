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
def is_repaired(self):
    """Check whether PDF was repaired."""
    pdf = _as_pdf_document(self)
    if not pdf:
        return False
    r = mupdf.pdf_was_repaired(pdf)
    if r:
        return True
    return False