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
def set_oc(self, oc=0):
    """Set / remove annotation OC xref."""
    CheckParent(self)
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    if not oc:
        mupdf.pdf_dict_del(annot_obj, PDF_NAME('OC'))
    else:
        JM_add_oc_object(mupdf.pdf_get_bound_document(annot_obj), annot_obj, oc)