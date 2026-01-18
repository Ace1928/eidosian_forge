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
def clear_with(self, value=None, bbox=None):
    """Fill all color components with same value."""
    if value is None:
        mupdf.fz_clear_pixmap(self.this)
    elif bbox is None:
        mupdf.fz_clear_pixmap_with_value(self.this, value)
    else:
        JM_clear_pixmap_rect_with_value(self.this, value, JM_irect_from_py(bbox))