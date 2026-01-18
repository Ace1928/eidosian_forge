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
def make_bookmark(self, loc):
    """Make a page pointer before layouting document."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    loc = mupdf.FzLocation(*loc)
    mark = mupdf.ll_fz_make_bookmark2(self.this.m_internal, loc.internal())
    return mark