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
@property
def is_external(self):
    if g_use_extra:
        return _extra.Outline_is_external(self.this)
    ol = self.this
    if not ol.m_internal:
        return False
    uri = ol.m_internal.uri if 1 else ol.uri()
    if uri is None:
        return False
    return mupdf.fz_is_external_link(uri)