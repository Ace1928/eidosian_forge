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
def store_shrink(percent):
    """
        Free 'percent' of current store size.
        """
    if percent >= 100:
        mupdf.fz_empty_store()
        return 0
    if percent > 0:
        mupdf.fz_shrink_store(100 - percent)