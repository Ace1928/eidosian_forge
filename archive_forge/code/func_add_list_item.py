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
def add_list_item(self):
    """Add item ("li" tag) under a (numbered or bulleted) list."""
    if self.tagname not in ('ol', 'ul'):
        raise ValueError('cannot add list item to', self.tagname)
    child = self.create_element('li')
    self.append_child(child)
    return child