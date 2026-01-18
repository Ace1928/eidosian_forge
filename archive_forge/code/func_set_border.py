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
def set_border(self, border=None, width=0, dashes=None, style=None):
    if type(border) is not dict:
        border = {'width': width, 'style': style, 'dashes': dashes}
    return self._setBorder(border, self.parent.parent.this, self.xref)