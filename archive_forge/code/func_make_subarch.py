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
def make_subarch(entries, mount, fmt):
    subarch = dict(fmt=fmt, entries=entries, path=mount)
    if fmt != 'tree' or self._subarchives == []:
        self._subarchives.append(subarch)
    else:
        ltree = self._subarchives[-1]
        if ltree['fmt'] != 'tree' or ltree['path'] != subarch['path']:
            self._subarchives.append(subarch)
        else:
            ltree['entries'].extend(subarch['entries'])
            self._subarchives[-1] = ltree