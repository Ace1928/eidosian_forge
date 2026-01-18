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
def apn_bbox(self):
    """annotation appearance bbox"""
    CheckParent(self)
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    ap = mupdf.pdf_dict_getl(annot_obj, PDF_NAME('AP'), PDF_NAME('N'))
    if not ap.m_internal:
        val = JM_py_from_rect(mupdf.FzRect(mupdf.FzRect.Fixed_INFINITE))
    else:
        rect = mupdf.pdf_dict_get_rect(ap, PDF_NAME('BBox'))
        val = JM_py_from_rect(rect)
    val = Rect(val) * self.get_parent().transformation_matrix
    val *= self.get_parent().derotation_matrix
    return val