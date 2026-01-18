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
def append_styled_span(self, style):
    span = self.create_element('span')
    span.add_style(style)
    prev = self.span_bottom()
    if prev is None:
        prev = self
    prev.append_child(span)
    return prev