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
def popup_rect(self):
    """annotation 'Popup' rectangle"""
    CheckParent(self)
    rect = mupdf.FzRect(mupdf.FzRect.Fixed_INFINITE)
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    obj = mupdf.pdf_dict_get(annot_obj, PDF_NAME('Popup'))
    if obj.m_internal:
        rect = mupdf.pdf_dict_get_rect(obj, PDF_NAME('Rect'))
    val = JM_py_from_rect(rect)
    val = Rect(val) * self.get_parent().transformation_matrix
    val *= self.get_parent().derotation_matrix
    return val