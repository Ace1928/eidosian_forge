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
def hdist(dir, a, b):
    dx = b.x - a.x
    dy = b.y - a.y
    return mupdf.fz_abs(dx * dir.x + dy * dir.y)