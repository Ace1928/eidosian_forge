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
def add_highlight_annot(self, quads=None, start=None, stop=None, clip=None) -> Annot:
    """Add a 'Highlight' annotation."""
    if quads is None:
        q = get_highlight_selection(self, start=start, stop=stop, clip=clip)
    else:
        q = CheckMarkerArg(quads)
    ret = self._add_text_marker(q, mupdf.PDF_ANNOT_HIGHLIGHT)
    return ret