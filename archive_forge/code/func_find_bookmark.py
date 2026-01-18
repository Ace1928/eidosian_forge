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
def find_bookmark(self, bm):
    """Find new location after layouting a document."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    location = mupdf.fz_lookup_bookmark2(self.this, bm)
    return (location.chapter, location.page)