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
def load_annot(self, ident: typing.Union[str, int]) -> Annot:
    """Load an annot by name (/NM key) or xref.

        Args:
            ident: identifier, either name (str) or xref (int).
        """
    CheckParent(self)
    if type(ident) is str:
        xref = 0
        name = ident
    elif type(ident) is int:
        xref = ident
        name = None
    else:
        raise ValueError('identifier must be a string or integer')
    val = self._load_annot(name, xref)
    if not val:
        return val
    val.thisown = True
    val.parent = weakref.proxy(self)
    self._annot_refs[id(val)] = val
    return val