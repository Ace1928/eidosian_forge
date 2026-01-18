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
def apn_matrix(self):
    """annotation appearance matrix"""
    try:
        CheckParent(self)
        annot = self.this
        assert isinstance(annot, mupdf.PdfAnnot)
        ap = mupdf.pdf_dict_getl(mupdf.pdf_annot_obj(annot), mupdf.PDF_ENUM_NAME_AP, mupdf.PDF_ENUM_NAME_N)
        if not ap.m_internal:
            return JM_py_from_matrix(mupdf.FzMatrix())
        mat = mupdf.pdf_dict_get_matrix(ap, mupdf.PDF_ENUM_NAME_Matrix)
        val = JM_py_from_matrix(mat)
        val = Matrix(val)
        return val
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        raise