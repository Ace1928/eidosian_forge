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
def can_save_incrementally(self):
    """Check whether incremental saves are possible."""
    pdf = _as_pdf_document(self)
    if not pdf:
        return False
    return mupdf.pdf_can_be_saved_incrementally(pdf)