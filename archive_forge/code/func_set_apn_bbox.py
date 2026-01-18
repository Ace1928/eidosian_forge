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
def set_apn_bbox(self, bbox):
    """
        Set annotation appearance bbox.
        """
    CheckParent(self)
    page = self.get_parent()
    rot = page.rotation_matrix
    mat = page.transformation_matrix
    bbox *= rot * ~mat
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    ap = mupdf.pdf_dict_getl(annot_obj, PDF_NAME('AP'), PDF_NAME('N'))
    if not ap.m_internal:
        raise RuntimeError(MSG_BAD_APN)
    rect = JM_rect_from_py(bbox)
    mupdf.pdf_dict_put_rect(ap, PDF_NAME('BBox'), rect)