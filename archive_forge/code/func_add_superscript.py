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
def add_superscript(self, text=None):
    """Add a superscript ("sup" tag)"""
    child = self.create_element('sup')
    if type(text) is str:
        child.append_child(self.create_text_node(text))
    prev = self.span_bottom()
    if prev is None:
        prev = self
    prev.append_child(child)
    return self