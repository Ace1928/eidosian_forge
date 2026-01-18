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
@staticmethod
def show_aa_level():
    """
        Show anti-aliasing values.
        """
    return dict(graphics=mupdf.fz_graphics_aa_level(), text=mupdf.fz_text_aa_level(), graphics_min_line_width=mupdf.fz_graphics_min_line_width())