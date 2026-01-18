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
@staticmethod
def set_graphics_min_line_width(min_line_width):
    """
        Set the graphics minimum line width.
        """
    mupdf.fz_set_graphics_min_line_width(min_line_width)