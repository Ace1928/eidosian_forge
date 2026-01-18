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
def annot_postprocess(page: 'Page', annot: 'Annot') -> None:
    """Clean up after annotation inertion.

    Set ownership flag and store annotation in page annotation dictionary.
    """
    assert isinstance(page, Page)
    assert isinstance(annot, Annot)
    annot.parent = page
    page._annot_refs[id(annot)] = annot
    annot.thisown = True