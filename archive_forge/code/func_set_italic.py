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
def set_italic(self, val=True):
    """Set italic on / off via CSS style"""
    if val:
        val = 'italic'
    else:
        val = 'normal'
    text = 'font-style: %s' % val
    self.append_styled_span(text)
    return self