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
def extractRAWDICT(self, cb=None, sort=False) -> dict:
    """Return page content as a Python dict of images and text characters."""
    val = self._textpage_dict(raw=True)
    if cb is not None:
        val['width'] = cb.width
        val['height'] = cb.height
    if sort is True:
        blocks = val['blocks']
        blocks.sort(key=lambda b: (b['bbox'][3], b['bbox'][0]))
        val['blocks'] = blocks
    return val