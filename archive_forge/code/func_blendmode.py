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
def blendmode(self):
    """annotation BlendMode"""
    CheckParent(self)
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    obj = mupdf.pdf_dict_get(annot_obj, PDF_NAME('BM'))
    blend_mode = None
    if obj.m_internal:
        blend_mode = JM_UnicodeFromStr(mupdf.pdf_to_name(obj))
        return blend_mode
    obj = mupdf.pdf_dict_getl(annot_obj, PDF_NAME('AP'), PDF_NAME('N'), PDF_NAME('Resources'), PDF_NAME('ExtGState'))
    if mupdf.pdf_is_dict(obj):
        n = mupdf.pdf_dict_len(obj)
        for i in range(n):
            obj1 = mupdf.pdf_dict_get_val(obj, i)
            if mupdf.pdf_is_dict(obj1):
                m = mupdf.pdf_dict_len(obj1)
                for j in range(m):
                    obj2 = mupdf.pdf_dict_get_key(obj1, j)
                    if mupdf.pdf_objcmp(obj2, PDF_NAME('BM')) == 0:
                        blend_mode = JM_UnicodeFromStr(mupdf.pdf_to_name(mupdf.pdf_dict_get_val(obj1, j)))
                        return blend_mode
    return blend_mode