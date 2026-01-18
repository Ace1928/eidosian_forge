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
def set_small_glyph_heights(on=None):
    """Set / unset small glyph heights."""
    global g_small_glyph_heights
    if on is not None:
        g_small_glyph_heights = bool(on)
        if g_use_extra:
            extra.set_small_glyph_heights(g_small_glyph_heights)
    return g_small_glyph_heights