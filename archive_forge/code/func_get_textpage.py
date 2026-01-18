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
def get_textpage(self, clip: rect_like=None, flags: int=0, matrix=None) -> 'TextPage':
    CheckParent(self)
    if matrix is None:
        matrix = Matrix(1, 1)
    old_rotation = self.rotation
    if old_rotation != 0:
        self.set_rotation(0)
    try:
        textpage = self._get_textpage(clip, flags=flags, matrix=matrix)
    finally:
        if old_rotation != 0:
            self.set_rotation(old_rotation)
    textpage = TextPage(textpage)
    textpage.parent = weakref.proxy(self)
    return textpage