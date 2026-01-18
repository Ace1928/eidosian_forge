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
def make_escape(ch):
    if ch == 92:
        return '\\u005c'
    elif 32 <= ch <= 127 or ch == 10:
        return chr(ch)
    elif 55296 <= ch <= 57343:
        return '\\ufffd'
    elif ch <= 65535:
        return '\\u%04x' % ch
    else:
        return '\\U%08x' % ch