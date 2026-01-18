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
def pil_save(self, *args, **kwargs):
    """Write to image file using Pillow.

        Args are passed to Pillow's Image.save method, see their documentation.
        Use instead of save when other output formats are desired.
        """
    try:
        from PIL import Image
    except ImportError:
        message('PIL/Pillow not installed')
        raise
    cspace = self.colorspace
    if cspace is None:
        mode = 'L'
    elif cspace.n == 1:
        mode = 'L' if self.alpha == 0 else 'LA'
    elif cspace.n == 3:
        mode = 'RGB' if self.alpha == 0 else 'RGBA'
    else:
        mode = 'CMYK'
    img = Image.frombytes(mode, (self.width, self.height), self.samples)
    if 'dpi' not in kwargs.keys():
        kwargs['dpi'] = (self.xres, self.yres)
    img.save(*args, **kwargs)