from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
def skip_bom(f):
    """
    Read past a BOM at the beginning of a source file.
    This could be added to the scanner, but it's *substantially* easier
    to keep it at this level.
    """
    if f.read(1) != u'\ufeff':
        f.seek(0)