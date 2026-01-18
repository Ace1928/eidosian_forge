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
def add_number_list(self, start=1, numtype=None):
    """Add numbered list ("ol" tag)"""
    child = self.create_element('ol')
    if start > 1:
        child.set_attribute('start', str(start))
    if numtype is not None:
        child.set_attribute('type', numtype)
    self.append_child(child)
    return child