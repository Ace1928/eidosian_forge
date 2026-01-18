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
def set_bgcolor(self, color):
    """Set background color via CSS style"""
    text = f'background-color: %s' % self.color_text(color)
    self.add_style(text)
    return self