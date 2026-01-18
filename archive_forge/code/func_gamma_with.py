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
def gamma_with(self, gamma):
    """Apply correction with some float.
        gamma=1 is a no-op."""
    if not mupdf.fz_pixmap_colorspace(self.this):
        JM_Warning('colorspace invalid for function')
        return
    mupdf.fz_gamma_pixmap(self.this, gamma)