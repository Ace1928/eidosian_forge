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
def paper_rect(s: str) -> Rect:
    """Return a Rect for the paper size indicated in string 's'. Must conform to the argument of method 'PaperSize', which will be invoked.
    """
    width, height = paper_size(s)
    return Rect(0.0, 0.0, width, height)