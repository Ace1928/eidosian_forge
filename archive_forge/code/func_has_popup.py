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
def has_popup(self):
    """Check if annotation has a Popup."""
    CheckParent(self)
    annot = self.this
    obj = mupdf.pdf_dict_get(mupdf.pdf_annot_obj(annot), PDF_NAME('Popup'))
    return True if obj.m_internal else False