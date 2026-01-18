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
def get_displaylist(self, annots=1):
    """
        Make a DisplayList from the page for Pixmap generation.

        Include (default) or exclude annotations.
        """
    CheckParent(self)
    if annots:
        dl = mupdf.fz_new_display_list_from_page(self.this)
    else:
        dl = mupdf.fz_new_display_list_from_page_contents(self.this)
    return DisplayList(dl)