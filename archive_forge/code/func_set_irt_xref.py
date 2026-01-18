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
def set_irt_xref(self, xref):
    """
        Set annotation IRT xref
        """
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    page = mupdf.pdf_annot_page(annot)
    if xref < 1 or xref >= mupdf.pdf_xref_len(page.doc()):
        raise ValueError(MSG_BAD_XREF)
    irt = mupdf.pdf_new_indirect(page.doc(), xref, 0)
    subt = mupdf.pdf_dict_get(irt, PDF_NAME('Subtype'))
    irt_subt = mupdf.pdf_annot_type_from_string(mupdf.pdf_to_name(subt))
    if irt_subt < 0:
        raise ValueError(MSG_IS_NO_ANNOT)
    mupdf.pdf_dict_put(annot_obj, PDF_NAME('IRT'), irt)