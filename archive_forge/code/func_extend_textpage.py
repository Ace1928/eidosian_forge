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
def extend_textpage(self, tpage, flags=0, matrix=None):
    page = self.this
    tp = tpage.this
    assert isinstance(tp, mupdf.FzStextPage)
    options = mupdf.FzStextOptions()
    options.flags = flags
    ctm = JM_matrix_from_py(matrix)
    dev = mupdf.FzDevice(tp, options)
    mupdf.fz_run_page(page, dev, ctm, mupdf.FzCookie())
    mupdf.fz_close_device(dev)