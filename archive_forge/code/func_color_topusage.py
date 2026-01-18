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
def color_topusage(self, clip=None):
    """Return most frequent color and its usage ratio."""
    allpixels = 0
    cnt = 0
    if clip is not None and self.irect in Rect(clip):
        clip = self.irect
    for pixel, count in self.color_count(colors=True, clip=clip).items():
        allpixels += count
        if count > cnt:
            cnt = count
            maxpixel = pixel
    if not allpixels:
        return (1, bytes([255] * self.n))
    return (cnt / allpixels, maxpixel)