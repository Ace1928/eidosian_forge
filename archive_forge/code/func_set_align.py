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
def set_align(self, align):
    """Set text alignment via CSS style"""
    text = 'text-align: %s'
    if isinstance(align, str):
        t = align
    elif align == TEXT_ALIGN_LEFT:
        t = 'left'
    elif align == TEXT_ALIGN_CENTER:
        t = 'center'
    elif align == TEXT_ALIGN_RIGHT:
        t = 'right'
    elif align == TEXT_ALIGN_JUSTIFY:
        t = 'justify'
    else:
        raise ValueError(f'Unrecognised align={align!r}')
    text = text % t
    self.add_style(text)
    return self