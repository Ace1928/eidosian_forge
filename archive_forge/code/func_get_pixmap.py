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
def get_pixmap(self, matrix=None, colorspace=None, alpha=0, clip=None):
    if isinstance(colorspace, Colorspace):
        colorspace = colorspace.this
    else:
        colorspace = mupdf.FzColorspace(mupdf.FzColorspace.Fixed_RGB)
    val = JM_pixmap_from_display_list(self.this, matrix, colorspace, alpha, clip, None)
    val.thisown = True
    return val