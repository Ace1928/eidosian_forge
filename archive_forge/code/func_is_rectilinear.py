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
@property
def is_rectilinear(self):
    """True if rectangles are mapped to rectangles."""
    return abs(self.b) < EPSILON and abs(self.c) < EPSILON or (abs(self.a) < EPSILON and abs(self.d) < EPSILON)