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
def repair_mono_font(page: 'Page', font: 'Font') -> None:
    """Repair character spacing for mono fonts.

    Notes:
        Some mono-spaced fonts are displayed with a too large character
        distance, e.g. "a b c" instead of "abc". This utility adds an entry
        "/W[0 65535 w]" to the descendent font(s) of font. The float w is
        taken to be the width of 0x20 (space).
        This should enforce viewers to use 'w' as the character width.

    Args:
        page: fitz.Page object.
        font: fitz.Font object.
    """
    if not font.flags['mono']:
        return None
    doc = page.parent
    fontlist = page.get_fonts()
    xrefs = [f[0] for f in fontlist if f[3] == font.name and f[4].startswith('F') and f[5].startswith('Identity')]
    if xrefs == []:
        return
    xrefs = set(xrefs)
    width = int(round(font.glyph_advance(32) * 1000))
    for xref in xrefs:
        if not TOOLS.set_font_width(doc, xref, width):
            log("Cannot set width for '%s' in xref %i" % (font.name, xref))