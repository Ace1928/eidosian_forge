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
def needs_pass(self):
    """Indicate password required."""
    if self.is_closed:
        raise ValueError('document closed')
    document = self.this if isinstance(self.this, mupdf.FzDocument) else self.this.super()
    ret = mupdf.fz_needs_password(document)
    return ret