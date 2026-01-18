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
def get_xy(arg):
    if isinstance(arg, (list, tuple)) and len(arg) == 2:
        return (arg[0], arg[1])
    if isinstance(arg, (Point, mupdf.FzPoint, mupdf.fz_point)):
        return (arg.x, arg.y)
    return (None, None)