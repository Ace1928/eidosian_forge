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
def extractTextbox(self, rect):
    this_tpage = self.this
    assert isinstance(this_tpage, mupdf.FzStextPage)
    area = JM_rect_from_py(rect)
    found = JM_copy_rectangle(this_tpage, area)
    rc = PyUnicode_DecodeRawUnicodeEscape(found)
    return rc