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
def adobe_glyph_unicodes() -> tuple:
    """
    Adobe Glyph List function
    """
    if _adobe_unicodes == {}:
        for line in _get_glyph_text():
            if line.startswith('#'):
                continue
            gname, unc = line.split(';')
            c = int('0x' + unc[:4], base=16)
            _adobe_unicodes[gname] = c
    return tuple(_adobe_unicodes.values())