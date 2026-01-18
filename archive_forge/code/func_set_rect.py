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
def set_rect(self, bbox, color):
    """Set color of all pixels in bbox."""
    pm = self.this
    n = pm.n()
    c = []
    for j in range(n):
        i = color[j]
        if not _INRANGE(i, 0, 255):
            raise ValueError(MSG_BAD_COLOR_SEQ)
        c.append(i)
    bbox = JM_irect_from_py(bbox)
    i = JM_fill_pixmap_rect_with_color(pm, c, bbox)
    rc = bool(i)
    return rc