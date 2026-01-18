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
def preshear(self, h, v):
    """Calculate pre shearing and replace current matrix."""
    h = float(h)
    v = float(v)
    a, b = (self.a, self.b)
    self.a += v * self.c
    self.b += v * self.d
    self.c += h * a
    self.d += h * b
    return self