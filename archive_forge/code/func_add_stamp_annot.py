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
def add_stamp_annot(self, rect: rect_like, stamp: int=0) -> Annot:
    """Add a ('rubber') 'Stamp' annotation."""
    old_rotation = annot_preprocess(self)
    try:
        annot = self._add_stamp_annot(rect, stamp)
    finally:
        if old_rotation != 0:
            self.set_rotation(old_rotation)
    annot_postprocess(self, annot)
    return annot