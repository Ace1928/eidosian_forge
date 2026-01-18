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
def last_location(self):
    """Id (chapter, page) of last page."""
    if self.is_closed:
        raise ValueError('document closed')
    last_loc = mupdf.fz_last_page(self.this)
    return (last_loc.chapter, last_loc.page)