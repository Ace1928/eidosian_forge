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
def add_line_annot(self, p1: point_like, p2: point_like) -> Annot:
    """Add a 'Line' annotation."""
    old_rotation = annot_preprocess(self)
    try:
        annot = self._add_line_annot(p1, p2)
    finally:
        if old_rotation != 0:
            self.set_rotation(old_rotation)
    annot_postprocess(self, annot)
    return annot