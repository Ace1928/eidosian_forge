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
def set_pagebreak_after(self):
    """Insert a page break after this node."""
    text = 'page-break-after: always'
    self.add_style(text)
    return self