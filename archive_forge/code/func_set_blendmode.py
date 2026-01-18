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
def set_blendmode(self, blend_mode):
    """Set annotation BlendMode."""
    CheckParent(self)
    annot = self.this
    annot_obj = mupdf.pdf_annot_obj(annot)
    mupdf.pdf_dict_put_name(annot_obj, PDF_NAME('BM'), blend_mode)