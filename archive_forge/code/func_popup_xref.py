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
def popup_xref(self):
    """annotation 'Popup' xref"""
    CheckParent(self)
    xref = 0
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    obj = mupdf.pdf_dict_get(annot_obj, PDF_NAME('Popup'))
    if obj.m_internal:
        xref = mupdf.pdf_to_num(obj)
    return xref